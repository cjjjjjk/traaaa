BASE_MODEL_VALUE = {
    "ALLOWED_CLASSES": ["car", "motorcycle", "bus", "truck"]
}

# Call frame camera
BASE_APICALL_VALUE = {
    "COOKIE_STR": """ASP.NET_SessionId=2pd400jxoiokdowndnju2zdr; .VDMS=E6D97CFABE16A23E4774230850C06957F36B5B3EA1FF8360A13AFBE6CB680692EF54AF6792F4900295E280829B09309FC1C42F3998D647739B7CDA12A3BBEC339828A6DCD87CE1E69EDE5FED4D54D3E641C6DE126BC213E650AF93CA7D237E3F0BBF1FF20C50762E6AE7BB049B7EE780F66A625A; _frontend=!DXIcY2WO+fUXfJrZrha5HPS1wJuimy99qr25sd/0cSvsQGa2NxADtaW2cwiwYrp8paE1JECWJWCs904=; CurrentLanguage=vi; _ga=GA1.3.4923489.1760691959; _gid=GA1.3.274225042.1760691959; _pk_id.1.2f14=7192ec68653c6b65.1760691959.1.1760691959.1760691959.1760691959.; _pk_ses.1.2f14=*; _ga_JCXT8BPG4E=GS2.3.s1760691960$o1$g0$t1760691960$j60$l0$h0; TS01e7700a=0150c7cfd1dc293c4c03a2ac85fec94a52acc6f241c039622d7f59470ce112b88ed79d54a48de2bf7d912f31e100bf2d5ad97a2cb8a0be33f0f7e148086b7a1d1cf1550652ae0a33c727506fdea7d7a8d673254c321f8ba65848425d90a7cb03f63360c739""",
    "HEADERS": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "vi,en;q=0.9",
        "Referer": "https://giaothong.hochiminhcity.gov.vn/"},
    "FRAME_BG": "black",
    "FRAME_W": "400",
    "FRAME_H": "260"
}

# Vehicle class 
VEHICLE_CLASS_COLORS = {
    "car": (255, 0, 0),         # Blue
    "motorcycle": (0, 255, 0),  # Green
    "bus": (0, 0, 255),         # Red
    "truck": (0, 255, 255)      # Yellow
}

# FIXED_VECTOR_LENGTHS = {
#     "car": 30,
#     "motorcycle": 20,
#     "bus": 40,
#     "truck": 40,
#     "default": 25 
# }