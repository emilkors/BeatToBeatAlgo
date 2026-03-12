import json

def get_message(index_path, ndjson_path, message_id):
    # load index
    with open(index_path, "r") as f:
        index = json.load(f)

    offset = index.get(message_id)
    if offset is None:
        return None

    # seek directly to message
    with open(ndjson_path, "rb") as f:
        f.seek(offset)
        line = f.readline()

    return json.loads(line)