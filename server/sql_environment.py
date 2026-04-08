# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SQL Repair Environment Implementation.

A pure MCP environment where an agent is given a broken SQL query and must
fix it. Correctness is evaluated by running both queries against an in-memory
SQLite database and comparing the result sets.

MCP tools:
- `get_task()`: Returns the current broken SQL, schema, and task description.
- `explore_db(sql)`: Execute a SELECT query for exploration (not graded).
- `submit_fix(sql)`: Runs the agent's SQL, compares to expected output, returns
  a score between 0.0 and 1.0.

Example:
    >>> from openenv.core.env_server.mcp_types import CallToolAction
    >>> env = SqlRepairEnvironment()
    >>> env.reset()
    >>>
    >>> obs = env.step(CallToolAction(tool_name="get_task", arguments={}))
    >>> print(obs.result)  # {"broken_sql": ..., "schema": ..., "description": ...}
    >>>
    >>> obs = env.step(CallToolAction(
    ...     tool_name="submit_fix",
    ...     arguments={"sql": "SELECT name FROM employees WHERE dept = 'eng'"}
    ... ))
    >>> print(obs.result)  # {"score": 1.0, "message": "Perfect match!", ...}
"""

import random
import re
import sqlite3
from collections import Counter
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Task definitions — 22 tasks across easy / medium / hard difficulties
# ---------------------------------------------------------------------------

# Each task is a dict with:
#   name         – unique identifier
#   difficulty   – "easy", "medium", or "hard"
#   schema_sql   – DDL + INSERT statements to set up the DB
#   broken_sql   – the query the agent must fix
#   expected_sql – the correct query used to compute the expected result set
#   description  – natural-language task description shown to the agent
#   schema_desc  – human-readable schema summary

TASKS = [
    # ======================================================================
    # EASY tasks (1-7): typos, missing WHERE, wrong operator
    # ======================================================================
    {
        "name": "easy_column_typo",
        "difficulty": "easy",
        "description": (
            "The query below has a typo in a column name. "
            "Fix the SQL so it returns all employee names from the engineering department."
        ),
        "schema_desc": (
            "Table: employees\n"
            "  id            INTEGER PRIMARY KEY\n"
            "  employee_name TEXT\n"
            "  dept          TEXT\n"
            "  salary        REAL"
        ),
        "schema_sql": """
            CREATE TABLE employees (
                id            INTEGER PRIMARY KEY,
                employee_name TEXT    NOT NULL,
                dept          TEXT    NOT NULL,
                salary        REAL    NOT NULL
            );
            INSERT INTO employees VALUES (1, 'Alice', 'engineering', 95000);
            INSERT INTO employees VALUES (2, 'Bob',   'engineering', 88000);
            INSERT INTO employees VALUES (3, 'Carol', 'marketing',   72000);
            INSERT INTO employees VALUES (4, 'Dave',  'engineering', 91000);
            INSERT INTO employees VALUES (5, 'Eve',   'marketing',   68000);
        """,
        "broken_sql": "SELECT emplyee_name FROM employees WHERE dept = 'engineering';",
        "expected_sql": "SELECT employee_name FROM employees WHERE dept = 'engineering';",
    },
    {
        "name": "easy_table_name_typo",
        "difficulty": "easy",
        "description": (
            "The query references a table name that does not exist. "
            "Fix the typo so it returns all products with price above 50."
        ),
        "schema_desc": (
            "Table: products\n"
            "  id     INTEGER PRIMARY KEY\n"
            "  name   TEXT\n"
            "  price  REAL\n"
            "  category TEXT"
        ),
        "schema_sql": """
            CREATE TABLE products (
                id       INTEGER PRIMARY KEY,
                name     TEXT NOT NULL,
                price    REAL NOT NULL,
                category TEXT NOT NULL
            );
            INSERT INTO products VALUES (1, 'Laptop',  999.99, 'electronics');
            INSERT INTO products VALUES (2, 'Pen',       2.50, 'office');
            INSERT INTO products VALUES (3, 'Monitor', 349.00, 'electronics');
            INSERT INTO products VALUES (4, 'Notebook',  5.99, 'office');
            INSERT INTO products VALUES (5, 'Keyboard', 79.99, 'electronics');
        """,
        "broken_sql": "SELECT name, price FROM prodcuts WHERE price > 50;",
        "expected_sql": "SELECT name, price FROM products WHERE price > 50;",
    },
    {
        "name": "easy_missing_where",
        "difficulty": "easy",
        "description": (
            "The query is supposed to return only active students, but it "
            "returns all students because the WHERE clause is missing. Add "
            "the missing filter."
        ),
        "schema_desc": (
            "Table: students\n"
            "  id      INTEGER PRIMARY KEY\n"
            "  name    TEXT\n"
            "  grade   TEXT\n"
            "  active  INTEGER (1 = active, 0 = inactive)"
        ),
        "schema_sql": """
            CREATE TABLE students (
                id     INTEGER PRIMARY KEY,
                name   TEXT NOT NULL,
                grade  TEXT NOT NULL,
                active INTEGER NOT NULL
            );
            INSERT INTO students VALUES (1, 'Alice',  'A', 1);
            INSERT INTO students VALUES (2, 'Bob',    'B', 0);
            INSERT INTO students VALUES (3, 'Carol',  'A', 1);
            INSERT INTO students VALUES (4, 'Dave',   'C', 0);
            INSERT INTO students VALUES (5, 'Eve',    'B', 1);
        """,
        "broken_sql": "SELECT name, grade FROM students;",
        "expected_sql": "SELECT name, grade FROM students WHERE active = 1;",
    },
    {
        "name": "easy_wrong_comparison_operator",
        "difficulty": "easy",
        "description": (
            "The query should find books published AFTER 2010 (not including 2010), "
            "but the comparison operator is wrong. Fix it."
        ),
        "schema_desc": (
            "Table: books\n"
            "  id      INTEGER PRIMARY KEY\n"
            "  title   TEXT\n"
            "  author  TEXT\n"
            "  year    INTEGER\n"
            "  rating  REAL"
        ),
        "schema_sql": """
            CREATE TABLE books (
                id     INTEGER PRIMARY KEY,
                title  TEXT NOT NULL,
                author TEXT NOT NULL,
                year   INTEGER NOT NULL,
                rating REAL NOT NULL
            );
            INSERT INTO books VALUES (1, 'Book A', 'Author 1', 2008, 4.2);
            INSERT INTO books VALUES (2, 'Book B', 'Author 2', 2010, 3.8);
            INSERT INTO books VALUES (3, 'Book C', 'Author 1', 2015, 4.5);
            INSERT INTO books VALUES (4, 'Book D', 'Author 3', 2020, 4.0);
            INSERT INTO books VALUES (5, 'Book E', 'Author 2', 2012, 3.5);
        """,
        "broken_sql": "SELECT title, author, year FROM books WHERE year >= 2010;",
        "expected_sql": "SELECT title, author, year FROM books WHERE year > 2010;",
    },
    {
        "name": "easy_wrong_string_literal",
        "difficulty": "easy",
        "description": (
            "The query filters on the wrong category value. It should find "
            "'electronics' items but uses 'electronic' (missing 's'). Fix the string."
        ),
        "schema_desc": (
            "Table: inventory\n"
            "  id        INTEGER PRIMARY KEY\n"
            "  item_name TEXT\n"
            "  category  TEXT\n"
            "  quantity  INTEGER"
        ),
        "schema_sql": """
            CREATE TABLE inventory (
                id        INTEGER PRIMARY KEY,
                item_name TEXT NOT NULL,
                category  TEXT NOT NULL,
                quantity  INTEGER NOT NULL
            );
            INSERT INTO inventory VALUES (1, 'Laptop',    'electronics', 50);
            INSERT INTO inventory VALUES (2, 'Desk',      'furniture',   30);
            INSERT INTO inventory VALUES (3, 'Monitor',   'electronics', 45);
            INSERT INTO inventory VALUES (4, 'Chair',     'furniture',   60);
            INSERT INTO inventory VALUES (5, 'Headphones','electronics', 100);
        """,
        "broken_sql": "SELECT item_name, quantity FROM inventory WHERE category = 'electronic';",
        "expected_sql": "SELECT item_name, quantity FROM inventory WHERE category = 'electronics';",
    },
    {
        "name": "easy_wrong_sort_direction",
        "difficulty": "easy",
        "description": (
            "The query should return the top 3 cities by population (highest first), "
            "but it sorts ascending instead of descending, returning the 3 smallest. "
            "Fix the ORDER BY direction."
        ),
        "schema_desc": (
            "Table: cities\n"
            "  id          INTEGER PRIMARY KEY\n"
            "  city_name   TEXT\n"
            "  country     TEXT\n"
            "  population  INTEGER"
        ),
        "schema_sql": """
            CREATE TABLE cities (
                id         INTEGER PRIMARY KEY,
                city_name  TEXT NOT NULL,
                country    TEXT NOT NULL,
                population INTEGER NOT NULL
            );
            INSERT INTO cities VALUES (1, 'Tokyo',     'Japan',  13960000);
            INSERT INTO cities VALUES (2, 'Delhi',     'India',  11030000);
            INSERT INTO cities VALUES (3, 'Shanghai',  'China',  24870000);
            INSERT INTO cities VALUES (4, 'Sao Paulo', 'Brazil', 12330000);
            INSERT INTO cities VALUES (5, 'Mumbai',    'India',  12480000);
        """,
        "broken_sql": "SELECT city_name, population FROM cities ORDER BY population ASC LIMIT 3;",
        "expected_sql": "SELECT city_name, population FROM cities ORDER BY population DESC LIMIT 3;",
    },
    {
        "name": "easy_select_star_instead_of_columns",
        "difficulty": "easy",
        "description": (
            "The query uses SELECT * but should only return the name and email "
            "columns for customers in the 'gold' tier. Fix the SELECT clause."
        ),
        "schema_desc": (
            "Table: customers\n"
            "  id     INTEGER PRIMARY KEY\n"
            "  name   TEXT\n"
            "  email  TEXT\n"
            "  tier   TEXT\n"
            "  balance REAL"
        ),
        "schema_sql": """
            CREATE TABLE customers (
                id      INTEGER PRIMARY KEY,
                name    TEXT NOT NULL,
                email   TEXT NOT NULL,
                tier    TEXT NOT NULL,
                balance REAL NOT NULL
            );
            INSERT INTO customers VALUES (1, 'Alice', 'alice@ex.com', 'gold',   500.0);
            INSERT INTO customers VALUES (2, 'Bob',   'bob@ex.com',   'silver', 200.0);
            INSERT INTO customers VALUES (3, 'Carol', 'carol@ex.com', 'gold',   750.0);
            INSERT INTO customers VALUES (4, 'Dave',  'dave@ex.com',  'bronze', 100.0);
        """,
        "broken_sql": "SELECT * FROM customers WHERE tier = 'gold';",
        "expected_sql": "SELECT name, email FROM customers WHERE tier = 'gold';",
    },
    # ======================================================================
    # MEDIUM tasks (8-14): wrong JOIN, missing GROUP BY, NULL handling,
    #                       wrong aggregate, subquery issues
    # ======================================================================
    {
        "name": "medium_wrong_join_column",
        "difficulty": "medium",
        "description": (
            "The query joins employees to departments but uses the wrong column "
            "in the ON clause. Fix it so every employee is paired with their "
            "correct department name."
        ),
        "schema_desc": (
            "Table: employees\n"
            "  id       INTEGER PRIMARY KEY\n"
            "  name     TEXT\n"
            "  dept_id  INTEGER  (foreign key -> departments.dept_id)\n"
            "  salary   REAL\n"
            "\n"
            "Table: departments\n"
            "  dept_id    INTEGER PRIMARY KEY\n"
            "  dept_name  TEXT\n"
            "  budget     REAL"
        ),
        "schema_sql": """
            CREATE TABLE departments (
                dept_id   INTEGER PRIMARY KEY,
                dept_name TEXT    NOT NULL,
                budget    REAL    NOT NULL
            );
            INSERT INTO departments VALUES (1, 'Engineering', 500000);
            INSERT INTO departments VALUES (2, 'Marketing',   300000);
            INSERT INTO departments VALUES (3, 'Sales',       200000);

            CREATE TABLE employees (
                id      INTEGER PRIMARY KEY,
                name    TEXT    NOT NULL,
                dept_id INTEGER NOT NULL REFERENCES departments(dept_id),
                salary  REAL    NOT NULL
            );
            INSERT INTO employees VALUES (1, 'Alice', 1, 95000);
            INSERT INTO employees VALUES (2, 'Bob',   2, 88000);
            INSERT INTO employees VALUES (3, 'Carol', 1, 72000);
            INSERT INTO employees VALUES (4, 'Dave',  3, 91000);
            INSERT INTO employees VALUES (5, 'Eve',   2, 68000);
        """,
        "broken_sql": (
            "SELECT e.name, d.dept_name "
            "FROM employees e "
            "JOIN departments d ON e.id = d.dept_id;"
        ),
        "expected_sql": (
            "SELECT e.name, d.dept_name "
            "FROM employees e "
            "JOIN departments d ON e.dept_id = d.dept_id;"
        ),
    },
    {
        "name": "medium_wrong_join_type",
        "difficulty": "medium",
        "description": (
            "The query uses INNER JOIN but should use LEFT JOIN. Some courses "
            "have no enrollments and should still appear with NULL student count. "
            "Fix the JOIN type."
        ),
        "schema_desc": (
            "Table: courses\n"
            "  id           INTEGER PRIMARY KEY\n"
            "  course_name  TEXT\n"
            "\n"
            "Table: enrollments\n"
            "  id          INTEGER PRIMARY KEY\n"
            "  course_id   INTEGER (foreign key -> courses.id)\n"
            "  student_name TEXT"
        ),
        "schema_sql": """
            CREATE TABLE courses (
                id          INTEGER PRIMARY KEY,
                course_name TEXT NOT NULL
            );
            INSERT INTO courses VALUES (1, 'Math 101');
            INSERT INTO courses VALUES (2, 'Physics 201');
            INSERT INTO courses VALUES (3, 'History 101');
            INSERT INTO courses VALUES (4, 'Art 101');

            CREATE TABLE enrollments (
                id           INTEGER PRIMARY KEY,
                course_id    INTEGER NOT NULL REFERENCES courses(id),
                student_name TEXT NOT NULL
            );
            INSERT INTO enrollments VALUES (1, 1, 'Alice');
            INSERT INTO enrollments VALUES (2, 1, 'Bob');
            INSERT INTO enrollments VALUES (3, 2, 'Carol');
            INSERT INTO enrollments VALUES (4, 3, 'Dave');
            INSERT INTO enrollments VALUES (5, 3, 'Eve');
        """,
        "broken_sql": (
            "SELECT c.course_name, COUNT(e.id) AS num_students "
            "FROM courses c "
            "JOIN enrollments e ON c.id = e.course_id "
            "GROUP BY c.course_name;"
        ),
        "expected_sql": (
            "SELECT c.course_name, COUNT(e.id) AS num_students "
            "FROM courses c "
            "LEFT JOIN enrollments e ON c.id = e.course_id "
            "GROUP BY c.course_name;"
        ),
    },
    {
        "name": "medium_null_handling",
        "difficulty": "medium",
        "description": (
            "The query should find all employees who do NOT have a manager "
            "(manager_id is NULL). Using = NULL does not work in SQL. Fix it."
        ),
        "schema_desc": (
            "Table: staff\n"
            "  id          INTEGER PRIMARY KEY\n"
            "  name        TEXT\n"
            "  manager_id  INTEGER (NULL for top-level managers)"
        ),
        "schema_sql": """
            CREATE TABLE staff (
                id         INTEGER PRIMARY KEY,
                name       TEXT NOT NULL,
                manager_id INTEGER
            );
            INSERT INTO staff VALUES (1, 'CEO Alice',  NULL);
            INSERT INTO staff VALUES (2, 'VP Bob',     1);
            INSERT INTO staff VALUES (3, 'VP Carol',   1);
            INSERT INTO staff VALUES (4, 'Mgr Dave',   2);
            INSERT INTO staff VALUES (5, 'CTO Eve',    NULL);
            INSERT INTO staff VALUES (6, 'Dev Frank',  4);
        """,
        "broken_sql": "SELECT name FROM staff WHERE manager_id = NULL;",
        "expected_sql": "SELECT name FROM staff WHERE manager_id IS NULL;",
    },
    {
        "name": "medium_wrong_aggregate",
        "difficulty": "medium",
        "description": (
            "The query should compute the average salary per department, "
            "but it uses SUM instead of AVG. Fix the aggregate function."
        ),
        "schema_desc": (
            "Table: payroll\n"
            "  id          INTEGER PRIMARY KEY\n"
            "  emp_name    TEXT\n"
            "  department  TEXT\n"
            "  salary      REAL"
        ),
        "schema_sql": """
            CREATE TABLE payroll (
                id         INTEGER PRIMARY KEY,
                emp_name   TEXT NOT NULL,
                department TEXT NOT NULL,
                salary     REAL NOT NULL
            );
            INSERT INTO payroll VALUES (1, 'Alice', 'Engineering', 95000);
            INSERT INTO payroll VALUES (2, 'Bob',   'Engineering', 88000);
            INSERT INTO payroll VALUES (3, 'Carol', 'Marketing',   72000);
            INSERT INTO payroll VALUES (4, 'Dave',  'Marketing',   68000);
            INSERT INTO payroll VALUES (5, 'Eve',   'Sales',       55000);
        """,
        "broken_sql": (
            "SELECT department, SUM(salary) AS avg_salary "
            "FROM payroll GROUP BY department;"
        ),
        "expected_sql": (
            "SELECT department, AVG(salary) AS avg_salary "
            "FROM payroll GROUP BY department;"
        ),
    },
    {
        "name": "medium_missing_group_by",
        "difficulty": "medium",
        "description": (
            "The query counts events per category but is missing the GROUP BY "
            "clause, so SQLite silently returns only one row with the total "
            "count instead of per-category counts. Add the GROUP BY."
        ),
        "schema_desc": (
            "Table: events\n"
            "  id        INTEGER PRIMARY KEY\n"
            "  category  TEXT\n"
            "  title     TEXT\n"
            "  attendees INTEGER"
        ),
        "schema_sql": """
            CREATE TABLE events (
                id        INTEGER PRIMARY KEY,
                category  TEXT NOT NULL,
                title     TEXT NOT NULL,
                attendees INTEGER NOT NULL
            );
            INSERT INTO events VALUES (1, 'music',  'Jazz Night',    120);
            INSERT INTO events VALUES (2, 'music',  'Rock Fest',     500);
            INSERT INTO events VALUES (3, 'sports', 'Marathon',     2000);
            INSERT INTO events VALUES (4, 'sports', 'Tennis Open',   800);
            INSERT INTO events VALUES (5, 'tech',   'Hackathon',     300);
            INSERT INTO events VALUES (6, 'tech',   'AI Summit',     450);
        """,
        "broken_sql": "SELECT category, COUNT(*) AS event_count FROM events;",
        "expected_sql": "SELECT category, COUNT(*) AS event_count FROM events GROUP BY category;",
    },
    {
        "name": "medium_incorrect_order_by_column",
        "difficulty": "medium",
        "description": (
            "The query should return the top 3 highest-rated restaurants, ordered "
            "by rating descending. But it orders by name instead of rating. "
            "Fix the ORDER BY column."
        ),
        "schema_desc": (
            "Table: restaurants\n"
            "  id      INTEGER PRIMARY KEY\n"
            "  name    TEXT\n"
            "  cuisine TEXT\n"
            "  rating  REAL\n"
            "  city    TEXT"
        ),
        "schema_sql": """
            CREATE TABLE restaurants (
                id      INTEGER PRIMARY KEY,
                name    TEXT NOT NULL,
                cuisine TEXT NOT NULL,
                rating  REAL NOT NULL,
                city    TEXT NOT NULL
            );
            INSERT INTO restaurants VALUES (1, 'Bella Italia', 'Italian',  4.5, 'NYC');
            INSERT INTO restaurants VALUES (2, 'Sushi Ko',     'Japanese', 4.8, 'NYC');
            INSERT INTO restaurants VALUES (3, 'Le Bistro',    'French',   4.2, 'Paris');
            INSERT INTO restaurants VALUES (4, 'Taco Loco',    'Mexican',  4.9, 'LA');
            INSERT INTO restaurants VALUES (5, 'Curry House',  'Indian',   4.6, 'London');
        """,
        "broken_sql": "SELECT name, rating FROM restaurants ORDER BY name DESC LIMIT 3;",
        "expected_sql": "SELECT name, rating FROM restaurants ORDER BY rating DESC LIMIT 3;",
    },
    # ======================================================================
    # HARD tasks (15-22): wrong HAVING, subquery errors, complex multi-bug,
    #                     window functions, CTEs
    # ======================================================================
    {
        "name": "hard_wrong_having_clause",
        "difficulty": "hard",
        "description": (
            "The query is supposed to find every customer whose total order "
            "amount exceeds $500, but the HAVING clause uses the wrong "
            "aggregate function. Fix it so it filters by total spend, not "
            "by order count."
        ),
        "schema_desc": (
            "Table: orders\n"
            "  id           INTEGER PRIMARY KEY\n"
            "  customer_id  INTEGER\n"
            "  product      TEXT\n"
            "  amount       REAL"
        ),
        "schema_sql": """
            CREATE TABLE orders (
                id          INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                product     TEXT    NOT NULL,
                amount      REAL    NOT NULL
            );
            INSERT INTO orders VALUES (1, 101, 'Widget',       200);
            INSERT INTO orders VALUES (2, 101, 'Widget',       350);
            INSERT INTO orders VALUES (3, 101, 'Gizmo',         75);
            INSERT INTO orders VALUES (4, 102, 'Gadget',       150);
            INSERT INTO orders VALUES (5, 102, 'Widget',       400);
            INSERT INTO orders VALUES (6, 102, 'Thingamajig',   25);
            INSERT INTO orders VALUES (7, 103, 'Doohickey',    100);
            INSERT INTO orders VALUES (8, 103, 'Gadget',        50);
            INSERT INTO orders VALUES (9, 104, 'Widget',       600);
        """,
        "broken_sql": (
            "SELECT customer_id, SUM(amount) AS total_spent "
            "FROM orders "
            "GROUP BY customer_id "
            "HAVING COUNT(*) > 2;"
        ),
        "expected_sql": (
            "SELECT customer_id, SUM(amount) AS total_spent "
            "FROM orders "
            "GROUP BY customer_id "
            "HAVING SUM(amount) > 500;"
        ),
    },
    {
        "name": "hard_subquery_wrong_column",
        "difficulty": "hard",
        "description": (
            "The query uses a subquery to find products whose price is above "
            "the average, but the subquery selects from the wrong column. "
            "Fix the subquery so it computes the average of price, not quantity."
        ),
        "schema_desc": (
            "Table: warehouse\n"
            "  id        INTEGER PRIMARY KEY\n"
            "  product   TEXT\n"
            "  price     REAL\n"
            "  quantity  INTEGER"
        ),
        "schema_sql": """
            CREATE TABLE warehouse (
                id       INTEGER PRIMARY KEY,
                product  TEXT NOT NULL,
                price    REAL NOT NULL,
                quantity INTEGER NOT NULL
            );
            INSERT INTO warehouse VALUES (1, 'Widget A',  25.00,  100);
            INSERT INTO warehouse VALUES (2, 'Widget B', 150.00,   20);
            INSERT INTO warehouse VALUES (3, 'Gadget C',  75.00,   50);
            INSERT INTO warehouse VALUES (4, 'Gadget D', 200.00,   10);
            INSERT INTO warehouse VALUES (5, 'Tool E',    50.00,   80);
        """,
        "broken_sql": (
            "SELECT product, price FROM warehouse "
            "WHERE price > (SELECT AVG(quantity) FROM warehouse);"
        ),
        "expected_sql": (
            "SELECT product, price FROM warehouse "
            "WHERE price > (SELECT AVG(price) FROM warehouse);"
        ),
    },
    {
        "name": "hard_missing_having_with_group_by",
        "difficulty": "hard",
        "description": (
            "The query should find authors who have written more than 1 book, "
            "showing author name and book count. It groups correctly but filters "
            "with WHERE instead of HAVING on the aggregate. Fix it."
        ),
        "schema_desc": (
            "Table: library\n"
            "  id      INTEGER PRIMARY KEY\n"
            "  title   TEXT\n"
            "  author  TEXT\n"
            "  genre   TEXT\n"
            "  pages   INTEGER"
        ),
        "schema_sql": """
            CREATE TABLE library (
                id     INTEGER PRIMARY KEY,
                title  TEXT NOT NULL,
                author TEXT NOT NULL,
                genre  TEXT NOT NULL,
                pages  INTEGER NOT NULL
            );
            INSERT INTO library VALUES (1, 'Book A', 'Smith',    'fiction',  300);
            INSERT INTO library VALUES (2, 'Book B', 'Smith',    'fiction',  250);
            INSERT INTO library VALUES (3, 'Book C', 'Johnson',  'science',  400);
            INSERT INTO library VALUES (4, 'Book D', 'Williams', 'history',  350);
            INSERT INTO library VALUES (5, 'Book E', 'Johnson',  'fiction',  200);
            INSERT INTO library VALUES (6, 'Book F', 'Williams', 'science',  500);
            INSERT INTO library VALUES (7, 'Book G', 'Smith',    'history',  450);
        """,
        "broken_sql": (
            "SELECT author, COUNT(*) AS book_count FROM library "
            "WHERE COUNT(*) > 1 GROUP BY author;"
        ),
        "expected_sql": (
            "SELECT author, COUNT(*) AS book_count FROM library "
            "GROUP BY author HAVING COUNT(*) > 1;"
        ),
    },
    {
        "name": "hard_multiple_bugs_join_and_filter",
        "difficulty": "hard",
        "description": (
            "This query has TWO bugs: (1) it uses RIGHT JOIN but the intended "
            "direction is a LEFT JOIN from accounts to transactions (all accounts, "
            "even those without transactions), and (2) the WHERE clause filters "
            "on t.amount > 0 but should filter on t.amount > 100. "
            "Fix both issues."
        ),
        "schema_desc": (
            "Table: accounts\n"
            "  id     INTEGER PRIMARY KEY\n"
            "  owner  TEXT\n"
            "  type   TEXT\n"
            "\n"
            "Table: transactions\n"
            "  id          INTEGER PRIMARY KEY\n"
            "  account_id  INTEGER (foreign key -> accounts.id)\n"
            "  amount      REAL\n"
            "  description TEXT"
        ),
        "schema_sql": """
            CREATE TABLE accounts (
                id    INTEGER PRIMARY KEY,
                owner TEXT NOT NULL,
                type  TEXT NOT NULL
            );
            INSERT INTO accounts VALUES (1, 'Alice', 'checking');
            INSERT INTO accounts VALUES (2, 'Bob',   'savings');
            INSERT INTO accounts VALUES (3, 'Carol', 'checking');

            CREATE TABLE transactions (
                id          INTEGER PRIMARY KEY,
                account_id  INTEGER NOT NULL REFERENCES accounts(id),
                amount      REAL NOT NULL,
                description TEXT NOT NULL
            );
            INSERT INTO transactions VALUES (1, 1,  250.00, 'Deposit');
            INSERT INTO transactions VALUES (2, 1,   50.00, 'Fee');
            INSERT INTO transactions VALUES (3, 2,  500.00, 'Deposit');
            INSERT INTO transactions VALUES (4, 2,   75.00, 'Withdrawal');
            INSERT INTO transactions VALUES (5, 3,  150.00, 'Deposit');
        """,
        "broken_sql": (
            "SELECT a.owner, t.amount, t.description "
            "FROM transactions t "
            "RIGHT JOIN accounts a ON t.account_id = a.id "
            "WHERE t.amount > 0;"
        ),
        "expected_sql": (
            "SELECT a.owner, t.amount, t.description "
            "FROM accounts a "
            "LEFT JOIN transactions t ON t.account_id = a.id "
            "WHERE t.amount > 100;"
        ),
    },
    {
        "name": "hard_distinct_missing",
        "difficulty": "hard",
        "description": (
            "The query should return unique city names where we have customers, "
            "but it returns duplicates. Also, the ORDER BY uses 'id' which is "
            "not in the SELECT list. Fix both: add DISTINCT and order by city."
        ),
        "schema_desc": (
            "Table: clients\n"
            "  id    INTEGER PRIMARY KEY\n"
            "  name  TEXT\n"
            "  city  TEXT\n"
            "  spent REAL"
        ),
        "schema_sql": """
            CREATE TABLE clients (
                id    INTEGER PRIMARY KEY,
                name  TEXT NOT NULL,
                city  TEXT NOT NULL,
                spent REAL NOT NULL
            );
            INSERT INTO clients VALUES (1, 'Alice', 'NYC',    500);
            INSERT INTO clients VALUES (2, 'Bob',   'LA',     300);
            INSERT INTO clients VALUES (3, 'Carol', 'NYC',    700);
            INSERT INTO clients VALUES (4, 'Dave',  'Chicago',200);
            INSERT INTO clients VALUES (5, 'Eve',   'LA',     400);
            INSERT INTO clients VALUES (6, 'Frank', 'Chicago',600);
        """,
        "broken_sql": "SELECT city FROM clients ORDER BY id;",
        "expected_sql": "SELECT DISTINCT city FROM clients ORDER BY city;",
    },
    {
        "name": "hard_union_wrong_column_count",
        "difficulty": "hard",
        "description": (
            "The query tries to combine active and archived project names with "
            "UNION, but the second SELECT includes an extra column (status) "
            "causing a column count mismatch. Also, it should use UNION ALL to "
            "keep duplicates. Fix both issues."
        ),
        "schema_desc": (
            "Table: active_projects\n"
            "  id    INTEGER PRIMARY KEY\n"
            "  name  TEXT\n"
            "  owner TEXT\n"
            "\n"
            "Table: archived_projects\n"
            "  id      INTEGER PRIMARY KEY\n"
            "  name    TEXT\n"
            "  owner   TEXT\n"
            "  status  TEXT"
        ),
        "schema_sql": """
            CREATE TABLE active_projects (
                id    INTEGER PRIMARY KEY,
                name  TEXT NOT NULL,
                owner TEXT NOT NULL
            );
            INSERT INTO active_projects VALUES (1, 'Alpha',   'Alice');
            INSERT INTO active_projects VALUES (2, 'Beta',    'Bob');

            CREATE TABLE archived_projects (
                id     INTEGER PRIMARY KEY,
                name   TEXT NOT NULL,
                owner  TEXT NOT NULL,
                status TEXT NOT NULL
            );
            INSERT INTO archived_projects VALUES (10, 'Gamma',  'Carol', 'completed');
            INSERT INTO archived_projects VALUES (11, 'Alpha',  'Dave',  'cancelled');
            INSERT INTO archived_projects VALUES (12, 'Delta',  'Eve',   'completed');
        """,
        "broken_sql": (
            "SELECT name, owner FROM active_projects "
            "UNION "
            "SELECT name, owner, status FROM archived_projects;"
        ),
        "expected_sql": (
            "SELECT name, owner FROM active_projects "
            "UNION ALL "
            "SELECT name, owner FROM archived_projects;"
        ),
    },
    {
        "name": "hard_nested_subquery_error",
        "difficulty": "hard",
        "description": (
            "The query uses a correlated subquery to find employees earning more "
            "than their department average. But the subquery incorrectly references "
            "the outer table alias, and uses MIN instead of AVG. Fix both issues."
        ),
        "schema_desc": (
            "Table: team\n"
            "  id          INTEGER PRIMARY KEY\n"
            "  emp_name    TEXT\n"
            "  department  TEXT\n"
            "  salary      REAL"
        ),
        "schema_sql": """
            CREATE TABLE team (
                id         INTEGER PRIMARY KEY,
                emp_name   TEXT NOT NULL,
                department TEXT NOT NULL,
                salary     REAL NOT NULL
            );
            INSERT INTO team VALUES (1, 'Alice', 'eng',   120000);
            INSERT INTO team VALUES (2, 'Bob',   'eng',    80000);
            INSERT INTO team VALUES (3, 'Carol', 'eng',    95000);
            INSERT INTO team VALUES (4, 'Dave',  'sales',  70000);
            INSERT INTO team VALUES (5, 'Eve',   'sales',  90000);
            INSERT INTO team VALUES (6, 'Frank', 'sales',  60000);
        """,
        "broken_sql": (
            "SELECT emp_name, department, salary FROM team t1 "
            "WHERE salary > (SELECT MIN(salary) FROM team t2 "
            "WHERE t2.department = t1.emp_name);"
        ),
        "expected_sql": (
            "SELECT emp_name, department, salary FROM team t1 "
            "WHERE salary > (SELECT AVG(salary) FROM team t2 "
            "WHERE t2.department = t1.department);"
        ),
    },
    {
        "name": "hard_window_function_ranking",
        "difficulty": "hard",
        "description": (
            "The query should rank employees within each department by salary "
            "(highest first), allowing ties to share the same rank. But it uses "
            "ROW_NUMBER (which never produces ties) instead of RANK, and it "
            "partitions by emp_name instead of department. Fix both: use RANK() "
            "and partition by department."
        ),
        "schema_desc": (
            "Table: salaries\n"
            "  id          INTEGER PRIMARY KEY\n"
            "  emp_name    TEXT\n"
            "  department  TEXT\n"
            "  salary      REAL"
        ),
        "schema_sql": """
            CREATE TABLE salaries (
                id         INTEGER PRIMARY KEY,
                emp_name   TEXT NOT NULL,
                department TEXT NOT NULL,
                salary     REAL NOT NULL
            );
            INSERT INTO salaries VALUES (1, 'Alice',  'eng',   120000);
            INSERT INTO salaries VALUES (2, 'Bob',    'eng',   120000);
            INSERT INTO salaries VALUES (3, 'Carol',  'eng',    95000);
            INSERT INTO salaries VALUES (4, 'Dave',   'sales',  90000);
            INSERT INTO salaries VALUES (5, 'Eve',    'sales',  90000);
            INSERT INTO salaries VALUES (6, 'Frank',  'sales',  70000);
            INSERT INTO salaries VALUES (7, 'Grace',  'eng',   110000);
        """,
        "broken_sql": (
            "SELECT emp_name, department, salary, "
            "ROW_NUMBER() OVER (PARTITION BY emp_name ORDER BY salary DESC) AS rnk "
            "FROM salaries;"
        ),
        "expected_sql": (
            "SELECT emp_name, department, salary, "
            "RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rnk "
            "FROM salaries;"
        ),
    },
    {
        "name": "hard_cte_self_join",
        "difficulty": "hard",
        "description": (
            "The query uses a CTE to compute each department's average salary, "
            "then joins back to find employees earning above their department "
            "average. But the CTE uses MIN instead of AVG, and the final "
            "comparison uses < instead of >. Fix both: compute AVG in the CTE "
            "and select employees whose salary is greater than the average."
        ),
        "schema_desc": (
            "Table: workforce\n"
            "  id          INTEGER PRIMARY KEY\n"
            "  emp_name    TEXT\n"
            "  department  TEXT\n"
            "  salary      REAL"
        ),
        "schema_sql": """
            CREATE TABLE workforce (
                id         INTEGER PRIMARY KEY,
                emp_name   TEXT NOT NULL,
                department TEXT NOT NULL,
                salary     REAL NOT NULL
            );
            INSERT INTO workforce VALUES (1, 'Alice',  'eng',   130000);
            INSERT INTO workforce VALUES (2, 'Bob',    'eng',    80000);
            INSERT INTO workforce VALUES (3, 'Carol',  'eng',    95000);
            INSERT INTO workforce VALUES (4, 'Dave',   'sales',  70000);
            INSERT INTO workforce VALUES (5, 'Eve',    'sales',  90000);
            INSERT INTO workforce VALUES (6, 'Frank',  'sales',  60000);
            INSERT INTO workforce VALUES (7, 'Grace',  'hr',     85000);
            INSERT INTO workforce VALUES (8, 'Hank',   'hr',     75000);
        """,
        "broken_sql": (
            "WITH dept_stats AS ("
            "  SELECT department, MIN(salary) AS dept_avg "
            "  FROM workforce GROUP BY department"
            ") "
            "SELECT w.emp_name, w.department, w.salary "
            "FROM workforce w "
            "JOIN dept_stats d ON w.department = d.department "
            "WHERE w.salary < d.dept_avg;"
        ),
        "expected_sql": (
            "WITH dept_stats AS ("
            "  SELECT department, AVG(salary) AS dept_avg "
            "  FROM workforce GROUP BY department"
            ") "
            "SELECT w.emp_name, w.department, w.salary "
            "FROM workforce w "
            "JOIN dept_stats d ON w.department = d.department "
            "WHERE w.salary > d.dept_avg;"
        ),
    },
]


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------


def _run_query(conn: sqlite3.Connection, sql: str):
    """Execute *sql* and return sorted list of rows, or raise on error."""
    cursor = conn.execute(sql)
    rows = cursor.fetchall()
    return sorted(rows)


def _score(
    agent_rows: list,
    expected_rows: list,
    *,
    is_syntax_error: bool = False,
) -> tuple[float, str]:
    """
    Compare result sets and return (score, message) on a continuous 0.0-1.0 scale.

    Scoring rubric (continuous):
      1.0  – exact match (same rows in any order)
      0.1  – syntax-valid query but completely wrong results (0 row overlap)
      0.0  – SQL syntax error (query didn't execute)

    For partial matches the score blends:
      - Row overlap:   what fraction of expected rows appear in agent output
      - Column match:  what fraction of column values match across shared rows
      - Row count penalty: extra/missing rows reduce score

    The final score is always clamped to [0.0, 1.0].
    """
    if is_syntax_error:
        return 0.01, "SQL syntax error — query did not execute."

    if agent_rows == expected_rows:
        return 0.99, "Perfect match! All rows and columns are correct."

    # Both empty — technically a match but edge case
    if not expected_rows and not agent_rows:
        return 0.99, "Both queries returned no rows — match."

    # Agent returned nothing but expected had rows
    if not agent_rows and expected_rows:
        return 0.1, (
            "Query executed successfully but returned no rows "
            f"(expected {len(expected_rows)})."
        )

    # Agent returned rows but expected was empty
    if agent_rows and not expected_rows:
        return 0.1, (
            f"Query returned {len(agent_rows)} rows but expected 0."
        )

    # --- Row overlap scoring ---
    # Use Counter (multiset) to correctly handle duplicate rows.
    expected_counter = Counter(expected_rows)
    agent_counter = Counter(agent_rows)
    # Multiset intersection: min count for each row present in both.
    overlap_counter = expected_counter & agent_counter
    overlap_count = sum(overlap_counter.values())

    row_overlap_score = overlap_count / max(len(expected_rows), 1)

    # --- Column-level scoring for non-overlapping rows ---
    # For rows that don't exactly match, check partial column matches
    col_score = 0.0
    # Subtract matched rows from each side to get remainders.
    non_overlap_expected_counter = expected_counter - overlap_counter
    non_overlap_agent_counter = agent_counter - overlap_counter
    non_overlap_expected = list(non_overlap_expected_counter.elements())
    non_overlap_agent = list(non_overlap_agent_counter.elements())

    if non_overlap_expected and non_overlap_agent:
        # Compare columns pairwise for best-match rows
        col_matches = 0
        col_total = 0
        for exp_row in non_overlap_expected:
            best_match = 0
            for agt_row in non_overlap_agent:
                if len(exp_row) == len(agt_row):
                    matches = sum(1 for a, b in zip(exp_row, agt_row) if a == b)
                    best_match = max(best_match, matches / len(exp_row))
                elif len(agt_row) > 0 and len(exp_row) > 0:
                    # Different column counts — partial credit based on overlap
                    min_cols = min(len(exp_row), len(agt_row))
                    matches = sum(
                        1 for a, b in zip(exp_row[:min_cols], agt_row[:min_cols])
                        if a == b
                    )
                    best_match = max(best_match, matches / max(len(exp_row), len(agt_row)))
            col_matches += best_match
            col_total += 1
        col_score = col_matches / col_total if col_total > 0 else 0.0

    # --- Row count penalty ---
    expected_count = len(expected_rows)
    agent_count = len(agent_rows)
    count_ratio = min(expected_count, agent_count) / max(expected_count, agent_count)

    # Blend: 60% row overlap + 25% column match on remainder + 15% count ratio
    # Minimum 0.1 for any query that runs without error
    raw = (
        0.60 * row_overlap_score
        + 0.25 * col_score
        + 0.15 * count_ratio
    )
    score = max(0.1, min(0.99, round(raw, 4)))

    # Build message
    parts = []
    if row_overlap_score > 0:
        parts.append(
            f"{overlap_count}/{len(expected_rows)} expected rows found in output"
        )
    else:
        parts.append("No expected rows found in output")
    parts.append(f"row count: got {agent_count}, expected {expected_count}")
    if col_score > 0:
        parts.append(f"partial column match: {col_score:.0%}")

    return score, ". ".join(parts) + "."


def _normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison: lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", sql.strip().lower())


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class SqlRepairEnvironment(MCPEnvironment):
    """
    SQL Repair MCP environment.

    The agent interacts exclusively through two MCP tools:
    - ``get_task``: fetch the broken SQL and schema for the current episode.
    - ``submit_fix``: submit a repaired SQL and receive a 0.0–1.0 score.

    A fresh in-memory SQLite database is created on every ``reset()``.
    """

    # Default maximum number of submit_fix calls per episode before forced
    # termination.  Can be overridden at construction time.
    DEFAULT_MAX_STEPS: int = 10

    def __init__(self, max_steps: int = DEFAULT_MAX_STEPS):
        """Initialise the environment with MCP server and tools.

        Args:
            max_steps: Maximum number of ``submit_fix`` calls allowed per
                episode.  After this many submissions the episode is marked
                done regardless of score.  Default is 10.
        """
        mcp = FastMCP("sql_repair_env")

        # ---- tool: get_task ------------------------------------------------
        @mcp.tool
        def get_task() -> dict:
            """
            Return the current task for the agent.

            Returns:
                Dictionary with keys:
                  - broken_sql (str): The SQL query that contains a bug.
                  - schema (str): Human-readable description of the database schema.
                  - description (str): Natural-language description of the task.
                  - difficulty (str): "easy", "medium", or "hard".
                  - task_name (str): Internal identifier for the active task.
                  - max_steps (int): Maximum submit_fix calls allowed.
                  - steps_used (int): Number of submissions so far.
            """
            task = self._current_task
            return {
                "broken_sql": task["broken_sql"],
                "schema": task["schema_desc"],
                "description": task["description"],
                "difficulty": task["difficulty"],
                "task_name": task["name"],
                "max_steps": self._max_steps,
                "steps_used": self._step_submissions,
            }

        # ---- tool: explore_db ------------------------------------------------
        @mcp.tool
        def explore_db(sql: str) -> dict:
            """
            Execute an arbitrary SELECT query against the database for
            exploration purposes. This does NOT count as a graded submission.

            Use this to inspect the schema, test hypotheses, or preview
            what the broken query returns before submitting a fix.

            Only SELECT statements are allowed. Results are capped at 50 rows.

            Args:
                sql: A SELECT query to execute.

            Returns:
                Dictionary with keys:
                  - columns (list[str]): Column names from the result.
                  - rows (list): Result rows (up to 50).
                  - row_count (int): Total number of rows returned.
                  - truncated (bool): True if results were capped at 50 rows.
                  - error (str | None): Error message if the query failed.
            """
            if self._done:
                return {
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "truncated": False,
                    "error": "Episode already finished.",
                }

            if self._conn is None:
                return {
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "truncated": False,
                    "error": "Environment not ready. Call reset() first.",
                }

            # Safety: only allow SELECT queries
            normalized = sql.strip().lower()
            forbidden = {"insert", "update", "delete", "drop", "alter", "create"}
            first_word = normalized.split()[0] if normalized.split() else ""
            if first_word in forbidden:
                return {
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "truncated": False,
                    "error": (
                        f"Only SELECT queries are allowed. "
                        f"Got: {first_word.upper()}"
                    ),
                }

            # Increment step count (but NOT _step_submissions)
            self._state.step_count += 1

            try:
                cursor = self._conn.execute(sql)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                all_rows = cursor.fetchall()
                row_count = len(all_rows)
                truncated = row_count > 50
                rows = [list(r) for r in all_rows[:50]]
                return {
                    "columns": columns,
                    "rows": rows,
                    "row_count": row_count,
                    "truncated": truncated,
                    "error": None,
                }
            except sqlite3.Error as exc:
                return {
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "truncated": False,
                    "error": f"SQL error: {exc}",
                }

        # ---- tool: submit_fix ----------------------------------------------
        @mcp.tool
        def submit_fix(sql: str) -> dict:
            """
            Submit a repaired SQL query to be evaluated.

            The query is executed against the in-memory database. The result set
            is compared to the expected output. A score between 0.0 and 1.0 is
            returned along with a human-readable message.

            The episode ends (done=True) when:
              - The agent achieves a perfect score (1.0), OR
              - The agent has used all allowed submissions (max_steps).

            Step-level reward shaping:
              - Improvement over the previous best score earns a small bonus.
              - SQL syntax errors receive 0.0 (lower than semantically wrong
                queries which get at least 0.1).

            Args:
                sql: The fixed SQL query to evaluate.

            Returns:
                Dictionary with keys:
                  - score (float): 0.0-1.0 continuous score.
                  - message (str): Explanation of the score.
                  - agent_rows (list): Rows returned by the submitted query.
                  - expected_rows (list): Rows returned by the expected query.
                  - done (bool): True if the episode is finished.
                  - steps_remaining (int): How many submissions are left.
                  - reward_delta (float): Change from previous best score
                    (positive = improvement, negative = regression).
            """
            # Guard: episode already finished
            if self._done:
                return {
                    "score": self._last_score,
                    "message": "Episode already finished.",
                    "agent_rows": [],
                    "expected_rows": self._expected_rows,
                    "done": True,
                    "steps_remaining": 0,
                    "reward_delta": 0.0,
                }

            if self._conn is None:
                return {
                    "score": 0.01,
                    "message": "Environment not ready. Call reset() before submit_fix().",
                    "agent_rows": [],
                    "expected_rows": [],
                    "done": False,
                    "steps_remaining": self._max_steps,
                    "reward_delta": 0.0,
                }

            # Guard: empty or whitespace-only SQL
            if not sql or not sql.strip():
                self._step_submissions += 1
                done = self._step_submissions >= self._max_steps
                if done:
                    self._done = True
                return {
                    "score": 0.01,
                    "message": "Empty SQL submission.",
                    "agent_rows": [],
                    "expected_rows": self._expected_rows,
                    "done": done,
                    "steps_remaining": max(0, self._max_steps - self._step_submissions),
                    "reward_delta": 0.0,
                }

            self._step_submissions += 1
            prev_best = self._best_score

            # Check if submission is identical to the broken query
            is_unchanged = (
                _normalize_sql(sql) == _normalize_sql(self._current_task["broken_sql"])
            )

            try:
                agent_rows = _run_query(self._conn, sql)
            except sqlite3.Error as exc:
                score = 0.01  # syntax error gets 0.01
                self._last_score = score
                done = (self._step_submissions >= self._max_steps)
                if done:
                    self._done = True
                return {
                    "score": score,
                    "message": f"SQL error: {exc}",
                    "agent_rows": [],
                    "expected_rows": self._expected_rows,
                    "done": done,
                    "steps_remaining": max(0, self._max_steps - self._step_submissions),
                    "reward_delta": score - prev_best,
                }

            raw_score, message = _score(agent_rows, self._expected_rows)

            # If the agent submitted the broken query unchanged, cap at 0.05
            if is_unchanged:
                score = min(raw_score, 0.05)
                message = (
                    "Submitted the original broken query unchanged. "
                    "Try to fix the bugs described in the task."
                )
            elif self._baseline_score < 1.0 and self._baseline_score > 0.0:
                # Baseline adjustment: subtract the broken query's free score
                # so improvements are measured relative to the broken baseline
                adjusted = max(0.0, (raw_score - self._baseline_score) / (1.0 - self._baseline_score))
                score = max(0.1, min(0.99, round(adjusted, 4)))
            else:
                score = raw_score

            # Clamp to strict (0, 1) — validator rejects 0.0 and 1.0
            score = max(0.01, min(0.99, score))

            self._last_score = score
            self._best_score = max(self._best_score, score)

            done = (score >= 0.99) or (self._step_submissions >= self._max_steps)
            if done:
                self._done = True
            reward_delta = score - prev_best

            return {
                "score": score,
                "message": message,
                "agent_rows": agent_rows,
                "expected_rows": self._expected_rows,
                "done": done,
                "steps_remaining": max(0, self._max_steps - self._step_submissions),
                "reward_delta": round(reward_delta, 4),
            }

        super().__init__(mcp)
        self._max_steps: int = max_steps
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task: dict = TASKS[0]
        self._conn: Optional[sqlite3.Connection] = None
        self._expected_rows: list = []
        self._last_score: float = 0.0
        self._best_score: float = 0.0
        self._step_submissions: int = 0
        self._rng = random.Random()

        # Initialize with a default episode so every instance is immediately
        # usable — required because the HTTP server pools env instances and
        # may route /step to a different instance than the one /reset hit.
        self._setup_episode(seed=None)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _setup_episode(self, seed: Optional[int]) -> None:
        """
        Select a task, build a fresh in-memory SQLite database, and
        pre-compute the expected result set.

        Called by both ``__init__`` (to guarantee a valid starting state)
        and ``reset()`` (to begin a new episode with a fresh task).
        """
        self._rng.seed(seed)
        self._current_task = self._rng.choice(TASKS)

        if self._conn is not None:
            self._conn.close()
        # check_same_thread=False is required: reset() and tool calls run
        # in different threads inside the OpenEnv async HTTP server.
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.executescript(self._current_task["schema_sql"])
        self._conn.commit()

        self._expected_rows = _run_query(self._conn, self._current_task["expected_sql"])
        self._last_score = 0.0
        self._best_score = 0.0
        self._step_submissions = 0
        self._done = False

        # Pre-compute baseline score of the broken query so we can adjust
        # all submissions relative to it (prevents free-riding on broken
        # queries that already score high).
        try:
            broken_rows = _run_query(self._conn, self._current_task["broken_sql"])
            baseline_raw, _ = _score(broken_rows, self._expected_rows)
        except sqlite3.Error:
            baseline_raw = 0.0
        self._baseline_score = baseline_raw

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment for a new episode.

        Creates a fresh in-memory SQLite database, populates it with the task
        schema, and pre-computes the expected result set.

        Args:
            seed: Optional random seed for reproducible task selection.
                  Pass the same seed to always get the same task difficulty.
                  None (default) picks a task at random each episode.
            episode_id: Optional episode ID; a UUID is generated if not provided.
            **kwargs: Additional reset options (ignored).

        Returns:
            Observation indicating the environment is ready, with task metadata.
        """
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._setup_episode(seed)

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "task_name": self._current_task["name"],
                "difficulty": self._current_task["difficulty"],
                "max_steps": self._max_steps,
                "message": (
                    "SQL Repair environment ready. "
                    "Call get_task() to see the broken query."
                ),
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Handle non-MCP actions with an informative error.

        This environment only supports MCP actions (ListToolsAction,
        CallToolAction). All other action types return an error observation.
        """
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": (
                    f"Unknown action type: {type(action).__name__}. "
                    "Use ListToolsAction or CallToolAction for MCP interactions."
                )
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute a step, incrementing the step counter and delegating to the
        base class for MCP routing.
        """
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Async step used by the WebSocket handler."""
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        """Return the current environment state."""
        return self._state
