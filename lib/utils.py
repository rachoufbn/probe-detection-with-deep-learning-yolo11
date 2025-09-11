import sys, os

def get_base_dir():
    """
    :return: path to project base directory
    """
    base_dir = os.path.abspath(
        os.path.join(__file__, "../..")
    )
    
    return base_dir

def get_input_folder_path():
    """
    fetches first command line argument,
    validates it is a valid folder and returns it

    :return: input folder path
    """
    if len(sys.argv) <= 1:
        raise Exception("Folder path not provided.")

    folder_path = sys.argv[1]

    if not os.path.isdir(folder_path):
        raise Exception("Invalid folder path.")
    
    return folder_path