from pathlib import Path
import pandas as pd


def identify_data_cursor_file_pairs(excel_destination, cursor_root, data_root):
    """
    Create an index mapping cursor files to matching data files.

    Handles:
    - .mat files
    - .index.json + .ndjson pairs
    """

    cursor_root = Path(cursor_root)
    data_root = Path(data_root)

    rows = []

    for subfolder in cursor_root.iterdir():
        if not subfolder.is_dir():
            continue

        subfolder_name = subfolder.name
        data_subfolder = data_root / subfolder_name

        if not data_subfolder.exists():
            continue

        # -------------------------
        # Build dataset lookup
        # -------------------------

        data_files = {}

        for f in data_subfolder.rglob("*"):
            if not f.is_file():
                continue

            name = f.name

            if name.endswith(".mat"):
                stem = f.stem
                data_files.setdefault(stem, {"mat": None, "index": None, "ndjson": None})
                data_files[stem]["mat"] = f

            elif name.endswith(".index.json"):
                stem = name.replace(".index.json", "")
                data_files.setdefault(stem, {"mat": None, "index": None, "ndjson": None})
                data_files[stem]["index"] = f

            elif name.endswith(".ndjson"):
                stem = f.stem
                data_files.setdefault(stem, {"mat": None, "index": None, "ndjson": None})
                data_files[stem]["ndjson"] = f

        # -------------------------
        # Cursor search
        # -------------------------

        subsubfolders = [f for f in subfolder.iterdir() if f.is_dir()]
        folders_to_search = subsubfolders if subsubfolders else [subfolder]

        for folder in folders_to_search:

            for cursor_file in folder.glob("*.json"):
                if not cursor_file.is_file():
                    continue

                # Case 1: flat structure
                if folder == subfolder:
                    cursor_name = cursor_file.stem
                    message_id = None

                # Case 2: nested structure
                else:
                    cursor_name = folder.name
                    message_id = cursor_file.stem

                dataset = data_files.get(cursor_name)

                if dataset:
                    rows.append({
                        "cursor_file_dest": str(cursor_file),
                        "data_mat_file": str(dataset["mat"]) if dataset["mat"] else None,
                        "data_index_json": str(dataset["index"]) if dataset["index"] else None,
                        "data_ndjson": str(dataset["ndjson"]) if dataset["ndjson"] else None,
                        "message_id": message_id,
                        "subfolder_name": subfolder_name
                    })

    df = pd.DataFrame(rows)

    df.to_excel(excel_destination, index=False)

    return df