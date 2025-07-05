import pickle

with open("certs/deriv1_certs.p", "rb") as f:
    cert_list = pickle.load(f)

entry = cert_list[0]

for key in entry:
    val = entry[key]
    print(f"{key}: type={type(val)}, shape={getattr(val, 'shape', 'N/A')}")
