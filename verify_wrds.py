"""Quick check that WRDS access works for JKP + CRSP."""
import wrds

c = wrds.Connection()

print("Testing JKP access...")
try:
    n = c.raw_sql(
        "SELECT COUNT(*) FROM contrib.global_factor "
        "WHERE excntry='USA' AND eom='2020-12-31'"
    ).iloc[0, 0]
    print(f"  JKP USA rows (Dec 2020): {n}")
except Exception as e:
    print(f"  JKP ERROR: {e}")

print("Testing CRSP access...")
try:
    n = c.raw_sql(
        "SELECT COUNT(*) FROM crsp.dsf WHERE date='2020-12-31'"
    ).iloc[0, 0]
    print(f"  CRSP daily rows (Dec 31 2020): {n}")
except Exception as e:
    print(f"  CRSP ERROR: {e}")

c.close()
print("Done.")
