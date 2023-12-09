import logging
import os
import sys
import zipfile
from enum import Enum
from typing import Any, Dict, List, Optional

os.system("python3 -m pip uninstall -y typing-extensions")
os.system("python3 -m pip install -U typing-extensions")
os.system(
    "python3 -m pip install -q --progress-bar off streamlink gradio tiktoken ultralytics pillow innertube opencv-python"
)
import cv2
import gradio as gr
import innertube
import numpy as np
import streamlink
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

model = YOLO("yolov8x.pt")


class SearchFilter(Enum):
    LIVE = ("EgJAAQ%3D%3D", "Live")
    VIDEO = ("EgIQAQ%3D%3D", "Video")

    def __init__(self, code, human_readable):
        self.code = code
        self.human_readable = human_readable

    def __str__(self):
        return self.human_readable


class SearchService:
    @staticmethod
    def search(
        query: Optional[str], filter: SearchFilter = SearchFilter.VIDEO
    ) -> (List[Dict[str, Any]], Optional[str]):
        client = innertube.InnerTube("WEB", "2.20230920.00.00")
        response = SearchService._search(query, filter)
        results = SearchService.parse(response)
        return results

    @staticmethod
    def parse(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        items = []

        contents = (
            data.get("contents", {})
            .get("twoColumnSearchResultsRenderer", {})
            .get("primaryContents", {})
            .get("sectionListRenderer", {})
            .get("contents", [])
        )
        if contents:
            items = contents[0].get("itemSectionRenderer", {}).get("contents", [])

        for item in items:
            if "videoRenderer" in item:
                renderer = item["videoRenderer"]
                video_id = renderer.get("videoId", "")
                thumbnail_urls = [
                    thumb.get("url", "")
                    for thumb in renderer.get("thumbnail", {}).get("thumbnails", [])
                ]
                title_text = "".join(
                    [
                        run.get("text", "")
                        for run in renderer.get("title", {}).get("runs", [])
                    ]
                )

                result = {
                    "video_id": video_id,
                    "thumbnail_urls": thumbnail_urls,
                    "title": title_text,
                }
                results.append(result)

        return results

    @staticmethod
    def _search(
        query: Optional[str] = None, filter: SearchFilter = SearchFilter.VIDEO
    ) -> Dict[str, Any]:
        client = innertube.InnerTube("WEB", "2.20230920.00.00")
        response = client.search(query=query, params=filter.code if filter else None)
        return response

    @staticmethod
    def get_youtube_url(video_id: str) -> str:
        return f"https://www.youtube.com/watch?v={video_id}"

    @staticmethod
    def get_stream_url(youtube_url):
        try:
            session = streamlink.Streamlink()
            streams = session.streams(youtube_url)
            if streams:
                best_stream = streams.get("best")
                return best_stream.url if best_stream else None
            else:
                logging.warning("No streams found for this URL")
                return None
        except Exception as e:
            logging.warning(f"An error occurred: {e}")
            return None


class LiveStreamAnnotator:
    def __init__(self):
        logging.getLogger().setLevel(logging.DEBUG)
        self.model = YOLO("yolov8x.pt")
        self.font_path = self.download_font(
            "https://www.fontsquirrel.com/fonts/download/open-sans",
            "open-sans.zip",
        )
        self.current_page_token = None
        self.streams = self.fetch_live_streams("world live cams")
        # Gradio UI Elements
        initial_gallery_items = [
            (stream["thumbnail_url"], stream["title"]) for stream in self.streams
        ]
        self.gallery = gr.Gallery(
            label="Live YouTube Videos",
            value=initial_gallery_items,
            show_label=False,
            columns=[3],
            rows=[10],
            object_fit="contain",
            height="auto",
        )
        self.search_input = gr.Textbox(label="Search Live YouTube Videos")
        self.stream_input = gr.Textbox(label="URL of Live YouTube Video")
        self.output_image = gr.AnnotatedImage(show_label=False)
        self.search_button = gr.Button("Search")
        self.submit_button = gr.Button("Detect Objects", variant="primary", size="lg")
        self.prev_page_button = gr.Button("Previous Page", interactive=False)
        self.next_page_button = gr.Button("Next Page", interactive=False)

    @staticmethod
    def download_font(url, save_path):
        os.system(f"wget {url} -O {save_path}")
        with zipfile.ZipFile(save_path, "r") as zip_ref:
            zip_ref.extractall(".")
        return os.path.join(".", "OpenSans-Regular.ttf")

    def capture_frame(self, url):
        stream_url = SearchService.get_stream_url(url)
        if not stream_url:
            return self.create_error_image("No stream found"), []
        frame = self.get_frame(stream_url)
        if frame is None:
            return self.create_error_image("Failed to capture frame"), []
        return self.process_frame(frame)

    def get_frame(self, stream_url):
        if not stream_url:
            return None
        try:
            cap = cv2.VideoCapture(stream_url)
            ret, frame = cap.read()
            cap.release()
            if ret:
                return cv2.resize(frame, (1920, 1080))
            else:
                logging.warning(
                    "Unable to process the HLS stream with cv2.VideoCapture."
                )
                return None
        except Exception as e:
            logging.warning(f"An error occurred while capturing the frame: {e}")
            return None

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb)
        annotations = self.get_annotations(results)
        return Image.fromarray(frame_rgb), annotations

    def get_annotations(self, results):
        annotations = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                bbox = tuple(map(int, box.xyxy[0]))
                annotations.append((bbox, class_name))
        return annotations

    @staticmethod
    def create_error_image(message):
        error_image = np.zeros((1920, 1080, 3), dtype=np.uint8)
        pil_image = Image.fromarray(error_image)
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.truetype("/usr/share/fonts/open-sans/OpenSans-Regular.ttf", 24)
        text_size = draw.textsize(message, font=font)
        position = ((1920 - text_size[0]) // 2, (1080 - text_size[1]) // 2)
        draw.text(position, message, (0, 0, 255), font=font)
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def fetch_live_streams(self, query=""):
        streams = []
        results = SearchService.search(
            query if query else "world live cams", SearchFilter.LIVE
        )
        for result in results:
            if "video_id" in result and "thumbnail_urls" in result:
                streams.append(
                    {
                        "thumbnail_url": result["thumbnail_urls"][0]
                        if result["thumbnail_urls"]
                        else "",
                        "title": result["title"],
                        "video_id": result["video_id"],
                        "label": result["video_id"],
                    }
                )
        return streams

    def render(self):
        with gr.Blocks(
            title="Object Detection in Live YouTube Streams",
            css="footer {visibility: hidden}",
        ) as app:
            gr.HTML(
                "<center><h1><b>Object Detection in Live YouTube Streams</b></h1></center>"
            )
            with gr.Column():
                self.stream_input.render()
                with gr.Group():
                    self.output_image.render()
                    self.submit_button.render()
            with gr.Group():
                with gr.Row():
                    self.search_input.render()
                    self.search_button.render()
                with gr.Row():
                    self.gallery.render()

            @self.gallery.select(
                inputs=None, outputs=[self.output_image, self.stream_input]
            )
            def on_gallery_select(evt: gr.SelectData):
                selected_index = evt.index
                if selected_index is not None and selected_index < len(self.streams):
                    selected_stream = self.streams[selected_index]
                    stream_url = SearchService.get_youtube_url(
                        selected_stream["video_id"]
                    )
                    frame_output = self.capture_frame(stream_url)
                    return frame_output, stream_url
                return None, ""

            @self.search_button.click(
                inputs=[self.search_input], outputs=[self.gallery]
            )
            def on_search_click(query):
                self.streams = self.fetch_live_streams(query)
                gallery_items = [
                    (stream["thumbnail_url"], stream["title"])
                    for stream in self.streams
                ]
                return gallery_items

            @self.submit_button.click(
                inputs=[self.stream_input], outputs=[self.output_image]
            )
            def annotate_stream(url):
                return self.capture_frame(url)

        app.queue().launch(show_api=False)

if __name__ == "__main__":
    LiveStreamAnnotator().render()
