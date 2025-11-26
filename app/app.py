from flask import Flask
#from routes.frames.stream_routes import stream_bp
#from routes.frames.map import map_bp
#from routes.camera.realtime_detect import realtime_detect_bp
from routes.camera.realtime_map import realtime_map_bp
from routes.camera.realtime_crawler import realtime_crawler_bp
# from routes.camera.realtime_analyze import realtime_analyze_bp
from routes.data.crud_frame_data import data_bp

app = Flask(__name__)
# app.register_blueprint(stream_bp, url_prefix="/frame")
# app.register_blueprint(map_bp, url_prefix="/frame")
# app.register_blueprint(realtime_detect_bp, url_prefix="/realtime")
app.register_blueprint(realtime_map_bp, url_prefix="/realtime")
app.register_blueprint(realtime_crawler_bp, url_prefix="/realtime")
# app.register_blueprint(realtime_analyze_bp, url_prefix="/realtime")

app.register_blueprint(data_bp, url_prefix="/")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
