# %%

# Make user-hashes?

raw_data['user'] = raw_data['user'].apply(hash)