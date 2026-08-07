"""
Run this directly to see the compressor in action:
    python example.py
"""

from compressor import SuperCompress  # relative import if run from inside this folder

# If you copied this folder into a project, use:
#   from supercompress_local.compressor import SuperCompress

CONTEXT = """
def fetch_user(user_id):
    row = db.query('SELECT * FROM users WHERE id=?', user_id)
    if row is None:
        return None
    return User(row)

User: my checkout keeps failing
Assistant: let me check the logs
Traceback (most recent call last):
  File "payments.py", line 55, in process_payment
    raise PaymentError('card declined')
PaymentError: card declined

INFO 2024-01-01T00:00:00 worker started
INFO 2024-01-01T00:00:01 worker started
INFO 2024-01-01T00:00:02 worker started
ERROR 2024-01-01T00:00:05 payment failed for order 123: card declined
DEBUG 2024-01-01T00:00:06 retry scheduled
"""

QUERY = "Why did checkout fail, and what does fetch_user return for a missing row?"

if __name__ == "__main__":
    sc = SuperCompress()
    result = sc.compress(CONTEXT, query=QUERY, budget_ratio=0.4)

    print(f"original tokens: {result.original_tokens}")
    print(f"kept tokens:     {result.kept_tokens}")
    print(f"savings:         {result.kv_savings_pct:.1f}%")
    print(f"preprocessor:    {result.preprocessor}")
    print(f"answer quality:  {result.answer_quality}")
    print()
    print("--- compressed text ---")
    print(result.compressed_text)
