import hashlib
def get_random_id_from_string(input_string):
    hash_object = hashlib.md5(input_string.encode())
    return str(hash_object.hexdigest())

