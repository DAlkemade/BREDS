import pickle

htmls_fname = 'htmls_VG_YOLO.pkl'
with open(htmls_fname, "rb") as f_html:
    htmls_lookup = pickle.load(f_html)
htmls_lookup = dict((k.lower(), v) for k, v in htmls_lookup.items())
with open(htmls_fname, 'wb') as f:
    pickle.dump(htmls_lookup, f, pickle.HIGHEST_PROTOCOL)