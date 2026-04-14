# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python (deepmlhub)
#     language: python
#     name: deepmlhub
# ---

# %%
# !pip install sqlalchemy -q

# %% [markdown]
# # Part 1: SQLAlchemy Core

# %% [markdown]
# # 1. Setup

# %%
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, ForeignKey, DateTime, text, select, insert, update, delete, func, join
from datetime import datetime

# %%
# Create an in-memory SQLite database (echo=True - for learning purposes)
engine = create_engine("sqlite:///:memory:", echo=True)

# %%
# Metadata is a container for Table objects
metadata = MetaData()

# %%
print("Engine created:", engine)

# %% [markdown]
# # 2. Define tables

# %%
# Define authors table
authors = Table(
    'authors', 
    metadata, 
    Column('id', Integer, primary_key=True),
    Column('name', String(100), nullable=False),
    Column('email', String(200), unique=True)
)

# Define articles table with foreign key
articles = Table(
    'articles', 
    metadata,
    Column('id', Integer, primary_key=True),
    Column('title', String(200), nullable=False),
    Column('content', String(5000)),
    Column('author_id', Integer, ForeignKey('authors.id')),
    Column('created_at', DateTime, default=datetime.utcnow)
)

# %%
# Create tables in database
metadata.create_all(engine)

# %%
print("Tables created!")
print(f"Columns in authors: {[c.name for c in authors.columns]}")
print(f"Columns in articles: {[c.name for c in articles.columns]}")

# %% [markdown]
# # 3. Insert Data

# %%
# Single insert
with engine.begin() as conn: 
    result = conn.execute(
        insert(authors).values(name="Alice Johnson", email = "alice@example.com")
    )
    print(f"Inserted author ID: {result.inserted_primary_key}")

# %%
# Bulk insert 
with engine.begin() as conn: 
    articles_data = [
        {"title": "Frist Post", "content": "Hello world!", "author_id": 1},
        {"title": "Second Post", "content": "More content", "author_id": 1},
        {"title": "Bob's Article", "content": "Writing from Bob.", "author_id": 2},
    ]
    result = conn.execute(insert(articles), articles_data)
    print(f"Inserted {result.rowcount} articles")

# %% [markdown]
# # 4. Select Data

# %%
# Select all authors
engine.echo = False # Turn off echo for cleaner output
with engine.connect() as conn: 
    result = conn.execute(select(authors))
    for row in result: 
        print(f"ID: {row.id}, Name: {row.name}, Email: {row.email}")

# %%
# Select withh WHERE clause
with engine.connect() as conn:
    stmt = select(articles).where(articles.c.author_id == 1)
    result = conn.execute(stmt)
    for row in result:
        print(row)
        print(f"{row.title}: {row.content[:30]}...")

# %% [markdown]
# # I did not insert bob and re-ran cells hence the error
# Perfect opportunity to learn

# %%
# Clearning existing data
with engine.begin() as conn: 
    conn.execute(delete(articles))
    conn.execute(delete(authors))
    print("Cleared existing data")

# %%
# Creating authors first
with engine.begin() as conn: 
    conn.execute(insert(authors).values(name = "Kriti Sanon", email="kritirobot@example.com" ))
    conn.execute(insert(authors).values(name = "Ananya Pandey", email="anayachunky@example.com"))
    print("Inserted 2 authors")

# %%
with engine.begin() as conn:
    articles_data = [
      {"title": "First Post", "content": "Hello world!", "author_id": 1},
      {"title": "Second Post", "content": "More content", "author_id": 1},
      {"title": "Pandey's Article", "content": "Writing from Ananya.", "author_id": 2},
      ]
    result = conn.execute(insert(articles), articles_data)
    print(f"Inserted {result.rowcount} articles")
    

# %%
engine.echo = False  # Turn off echo for cleaner output
with engine.connect() as conn:
  result = conn.execute(select(authors))
  for row in result:
      print(f"ID: {row.id}, Name: {row.name}, Email: {row.email}")

# %%
with engine.connect() as conn:
    stmt = select(articles).where(articles.c.author_id == 1)
    result = conn.execute(stmt)
    for row in result:
        print(f"{row.title}: {row.content[:30]}...")

# %% [markdown]
# # 5. JOIN Operations

# %%
# Joing articles with authors
with engine.connect() as conn: 
    stmt = (
        select(articles.c.title, authors.c.name.label("author_name"))
        .select_from(articles.join(authors, articles.c.author_id == authors.c.id))
    )
    result = conn.execute(stmt)
    for row in result: 
        print(f"{row.title} by {row.author_name}")

# %% [markdown]
# # 6. Aggregations

# %%
# Count articles per author
with engine.connect() as conn: 
    stmt = (
    select(authors.c.name, func.count(articles.c.id).label("count"))
    .select_from(authors.join(articles, authors.c.id == articles.c.author_id))
    .group_by(authors.c.id, authors.c.name)
    )
    result = conn.execute(stmt)
    for row in result: 
        print(f"{row.name} : {row.count} articles")

# %% [markdown]
# # 7. Update Data

# %%
with engine.begin() as conn: 
    stmt = (
        update(authors)
        .where(authors.c.name == "Kriti Sanon")
        .values(email = "newKriti@example.com")
    )
    result = conn.execute(stmt)
    print(f"Updated {result.rowcount} rows")

# %%

# %% [markdown]
# -------------------------------------------------------------------------------------------------------------------

# %%

# %% [markdown]
# # Part 2. SQLAlchemy ORM

# %% [markdown]
# # 8. ORM Setup

# %%
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session, joinedload, selectinload
from typing import List

# %%
# New database for ORM
orm_engine = create_engine("sqlite:///:memory:", echo=False)


# %%
class Base(DeclarativeBase): 
    pass


# %% [markdown]
# # 9. Define ORM Models

# %%
class Customer(Base): 
    __tablename__ = 'customers'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(200), unique=True)
        
    orders: Mapped[List["Order"]] = relationship(
            back_populates="customer", cascade="all, delete-orphan"
    )
        
    
    def __repr__(self): 
        return f"<Customer(id={self.id}, name='{self.name}')>"
    

# %%
class Product(Base): 
    __tablename__ = 'products'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    price: Mapped[float] = mapped_column(Integer)

    order_items: Mapped[List["OrderItem"]] = relationship(back_populates="product")

    def __repr__(self):
        return f"<Product(id={self.id}, name='{self.name}', price={self.price})>"


# %%
class Order(Base):
    __tablename__ = 'orders'
    __table_args__ = {'extend_existing': True}

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey('customers.id'))
    status: Mapped[str] = mapped_column(String(50), default="pending")

    customer: Mapped["Customer"] = relationship(back_populates="orders")
    items: Mapped[List["OrderItem"]] = relationship(
      back_populates="order", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Order(id={self.id}, status='{self.status}')>"



# %%
class OrderItem(Base):
    __tablename__ = 'order_items'
    __table_args__ = {'extend_existing': True}

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey('orders.id'))
    product_id: Mapped[int] = mapped_column(ForeignKey('products.id'))
    quantity: Mapped[int] = mapped_column(Integer)

    order: Mapped["Order"] = relationship(back_populates="items")
    product: Mapped["Product"] = relationship(back_populates="order_items")
        
    def __repr__(self):
        return f"<OrderItem(id={self.id}, quantity={self.quantity})>"


# %% [markdown]
# ### Lesson

# %% [raw]
# __repr__ controls what you see when you print/log an object. Without it, you get unhelpful output like <__main__.OrderItem object at 0x10a3b5c70>.
#
#
# With repr:
# print(order_item)  # <OrderItem(id=1, quantity=2)>
# Without repr:
# print(order_item)  # <__main__.OrderItem object at 0x10a3b5c70>

# %%
# Create tables
Base.metadata.create_all(orm_engine)
print("ORM Tables created")

# %% [markdown]
# # 10. Create Objects

# %%
session = Session(orm_engine)

# %%
# Create products and customers
laptop = Product(name="Laptop", price=999)
mouse = Product(name = "Mouse", price=29)
john = Customer(name = "John Doe", email="john@example.com")

# %%
session.add_all([laptop, mouse, john])
session.commit()

# %% [markdown]
# # 11. Work with Relationships

# %%
# Create an order with items
order  = Order(
    customer=john, 
    items = [
        OrderItem(product=laptop, quantity=1), 
        OrderItem(product=mouse, quantity=2),
    ]
    )
session.add(order)
session.commit()

# %% [markdown]
# # 12. Query - Lazy vs Eager Loading (supposedly very important)

# %%
# Lazy Loading (N+1 problem)
orm_engine.echo = True
orders = session.query(Order).all()
for order in orders: 
    print(f"Order : {order.id} by {order.customer.name}") # Triggers extra query per order
orm_engine.echo = False

# %%
# EAGER loading - single query 
orm_engine.echo = True
orders = session.query(Order).options(joinedload(Order.customer)).all()
for order in orders: 
    print(f"Order : {order.id} by {order.customer.name}") # No extra queries
orm_engine.echo = False

# %% [raw]
# # Lessson
#
# The difference is when the related customer data gets fetched:
# Lazy Loading (N+1 problem)
# orders = session.query(Order).all()  # Query 1: SELECT orders...
# for order in orders:
#     print(order.customer.name)       # Query 2,3,4...: SELECT customers... for EACH order
# - Runs 1 + N queries (1 for orders, N for each customer)
# - Good when you don't need the related data
# - Terrible for loops - each .customer.name triggers a new database hit
# Eager Loading (joinedload)
# orders = session.query(Order).options(joinedload(Order.customer)).all()
# # Single query with JOIN: SELECT orders... LEFT OUTER JOIN customers...
# for order in orders:
#     print(order.customer.name)       # Already loaded, no extra query
# - Runs 1 query total using SQL JOIN
# - Good when you know you'll use the related data
# - No N+1 penalty in loops
# The Rule: Use joinedload() when you'll access the relationship in a loop. Use lazy loading when you might not need the related data at all.

# %% [markdown]
# # 13. Update and Delete

# %%
laptop = session.query(Product).filter_by(name="Laptop").first()
laptop.price = 899 
session.commit()
print(f"New Laptop price : {laptop.price}")

# %%
# Delete
item_to_delete = session.query(OrderItem).first()
session.delete(item_to_delete)
session.commit()
print("Item deleted")

# %% [markdown]
# # 14. Aggregation

# %%
from sqlalchemy import func

results = session.query(
    Customer.name, 
    func.count(Order.id).label('order_count')
).join(Order).group_by(Customer.id).all()

for name, count in results:
    print(f"{name} : {count} orders")

# %% [markdown]
# ----------------------------------------------------------------------------------------------------------------

# %%

# %% [markdown]
# # Part 3: Core vs ORM side-by-side

# %%
# Same operation: select with filter

# %%
# CORE 
with engine.connect() as conn: 
    stmt = select(authors).where(authors.c.name.like("k%"))
    for rown in conn.execute(stmt): 
        print(f"CORE: {row.name}")
        
# ORM 
for customer in session.query(Customer).filter(Customer.name.like("A%")): 
    print(f"ORM : {customer.name}")

# %%

# %% [markdown]
# # Lessons

# %% [raw]
# Core and ORM deal with data operations.
# Alembic deals with schema evolution.

# %% [raw]
# In serious systems:
#
# 80% ORM
#
# 15% Core for heavy queries
#
# 5% raw SQL for extreme cases
#
# Alembic always
#
# If someone uses only ORM and never reads generated SQL,
# they’re flying blind.
#
# If someone refuses ORM entirely,
# they’re wasting time on CRUD.
#
# Balance is maturity.

# %%
