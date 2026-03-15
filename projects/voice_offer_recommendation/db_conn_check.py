from supabase_client import supabase

# 1. Get all products (limit 5)
result = supabase.table("products").select("*").limit(5).execute()
print(f"Products: {len(result.data)}")
for p in result.data[:2]:
    print(f"  - {p['name']}: ${p['price']}")

# 2. Get coupon usage data
result = supabase.table("coupon_usage").select("*").limit(5).execute()
print(f"\nCoupon usage records: {len(result.data)}")
for record in result.data[:3]:
    print(f"  - User {record.get('user_id', 'N/A')}: {record}")
