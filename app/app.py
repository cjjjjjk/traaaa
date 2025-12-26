from flask import Flask
#from routes.frames.stream_routes import stream_bp
#from routes.frames.map import map_bp
#from routes.camera.realtime_detect import realtime_detect_bp
from routes.camera.realtime_map import realtime_map_bp
from routes.camera.realtime_crawler import realtime_crawler_bp
from routes.camera.local_crawler import local_crawler_bp
from routes.camera.realtime_model_apply import realtime_model_apply_bp
# from routes.camera.realtime_analyze import realtime_analyze_bp
from routes.data.crud_frame_data import data_bp
from routes.data.fix_filenames import fix_filenames_bp
from routes.labeled.auto_labeled import auto_labeled_bp
from routes.data.sync_sheet import sync_sheet_bp
from routes.model.create_model import create_model_bp


app = Flask(__name__)
app.register_blueprint(realtime_map_bp, url_prefix="/realtime")
app.register_blueprint(realtime_crawler_bp, url_prefix="/realtime")
app.register_blueprint(local_crawler_bp, url_prefix="/realtime")
app.register_blueprint(realtime_model_apply_bp, url_prefix="/realtime")

app.register_blueprint(data_bp, url_prefix="/")
app.register_blueprint(fix_filenames_bp, url_prefix="/data")
app.register_blueprint(auto_labeled_bp, url_prefix="/labeled")

app.register_blueprint(sync_sheet_bp, url_prefix="/data")
app.register_blueprint(create_model_bp, url_prefix="/model")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
