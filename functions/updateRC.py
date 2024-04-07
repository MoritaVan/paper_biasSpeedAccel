def fontsizeDict(small:int=10, medium:int=12):
    SMALL_SIZE = small
    MEDIUM_SIZE = medium

    return {"font.size":SMALL_SIZE,
            "axes.titlesize":MEDIUM_SIZE,
            "axes.labelseize":MEDIUM_SIZE,
            "xtick.labelsize":SMALL_SIZE,
            "ytick.labelsize":SMALL_SIZE,
            "legend.fontsize":SMALL_SIZE,
            "figure.titlesize":MEDIUM_SIZE
            }

def rcConfigDict(filepath:str):
    import json

    with open(filepath) as f:
        json_s = f.read()
        rcDict = json.loads(json_s)
    
    return rcDict