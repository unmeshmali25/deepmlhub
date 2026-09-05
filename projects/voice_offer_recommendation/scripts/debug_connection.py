import os
from supabase import create_client

# Print what's actually in environment
print("=" * 60)
print("ENVIRONMENT VARIABLES CHECK")
print("=" * 60)
print(f"SUPABASE_URL: {os.getenv('SUPABASE_URL', 'NOT SET')}")
print(f"SUPABASE_KEY: {'SET' if os.getenv('SUPABASE_KEY') else 'NOT SET'}")
print(
    f"SUPABASE_SERVICE_KEY: {'SET' if os.getenv('SUPABASE_SERVICE_KEY') else 'NOT SET'}"
)
print(
    f"DATABASE_URL: {os.getenv('DATABASE_URL', 'NOT SET')[:50]}..."
    if os.getenv("DATABASE_URL")
    else "DATABASE_URL: NOT SET"
)
# Check if .env file exists and what it contains
print("\n" + "=" * 60)
print("CHECKING .env FILES")
print("=" * 60)
for env_file in [".env", ".env.production", ".env.local"]:
    if os.path.exists(env_file):
        print(f"\n✅ {env_file} exists")
        with open(env_file, "r") as f:
            content = f.read()
            if "SUPABASE_URL" in content:
                # Extract just the URL line
                for line in content.split("\n"):
                    if line.startswith("SUPABASE_URL="):
                        print(f"   Found: {line}")
                        break
    else:
        print(f"\n❌ {env_file} not found")
# Try to connect and verify
print("\n" + "=" * 60)
print("CONNECTION TEST")
print("=" * 60)
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
if supabase_url and supabase_key:
    try:
        supabase = create_client(supabase_url, supabase_key)
        print(f"✅ Connected to: {supabase_url}")

        # Try to query orders
        print("\n📦 Querying orders table...")
        result = supabase.table("orders").select("*", count="exact").limit(1).execute()
        print(f"   Result count: {result.count if hasattr(result, 'count') else 'N/A'}")
        print(f"   Data length: {len(result.data)}")
        if result.data:
            print(f"   Sample order ID: {result.data[0].get('id', 'N/A')}")

        # Try to query users
        print("\n👥 Querying users table...")
        result = supabase.table("users").select("*", count="exact").limit(1).execute()
        print(f"   Result count: {result.count if hasattr(result, 'count') else 'N/A'}")
        print(f"   Data length: {len(result.data)}")

        # Try to query shopping_sessions
        print("\n🛒 Querying shopping_sessions table...")
        result = (
            supabase.table("shopping_sessions")
            .select("*", count="exact")
            .limit(1)
            .execute()
        )
        print(f"   Result count: {result.count if hasattr(result, 'count') else 'N/A'}")
        print(f"   Data length: {len(result.data)}")

        # Try to query agents (should have data)
        print("\n🤖 Querying agents table...")
        result = supabase.table("agents").select("*", count="exact").limit(1).execute()
        print(f"   Result count: {result.count if hasattr(result, 'count') else 'N/A'}")
        print(f"   Data length: {len(result.data)}")
        if result.data:
            print(f"   Sample agent ID: {result.data[0].get('agent_id', 'N/A')}")

        # Try to query products (should have data)
        print("\n📦 Querying products table...")
        result = (
            supabase.table("products").select("*", count="exact").limit(1).execute()
        )
        print(f"   Result count: {result.count if hasattr(result, 'count') else 'N/A'}")
        print(f"   Data length: {len(result.data)}")

    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ Missing credentials - cannot connect")
print("\n" + "=" * 60)
print("TROUBLESHOOTING")
print("=" * 60)
print("""
If you're seeing 0 records in orders/users but have agents/products:
1. Check you're using the SAME credentials as VoiceOffers repo
2. The production URL should be: https://xrzpyapwnygdcjmcmnxg.supabase.co
3. Try: rm -rf __pycache__ && unset SUPABASE_URL && python debug_connection.py
4. Make sure you're loading the .env file BEFORE importing supabase_client
""")
