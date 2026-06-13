# pyrefly: ignore [missing-import]
from roboflow import Roboflow
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))


# workspace = rf.workspace("maedakenji")
# workspace.deploy_model(
#     model_type = "yolo26",
#     model_path = r"D:\Code\Python\brin\output\20260403_124022_425303",
#     # filename="weights/best.pt",
#     project_ids = ["surrounding-awareness"],
#     model_name="my-custom-model"
# )

project = rf.workspace().project("surrounding-awareness")
version = project.version(5)
version.deploy("yolo26", r"D:\Code\Python\brin\output\20260403_201314_767735")