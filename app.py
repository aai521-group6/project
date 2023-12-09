import logging
import sys
from enum import Enum
from typing import Any, Dict, Optional

import cv2
import gradio as gr
import innertube
import numpy as np
import streamlink
from PIL import Image
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
    def search(query: Optional[str], filter: SearchFilter = SearchFilter.VIDEO):
        client = innertube.InnerTube("WEB", "2.20230920.00.00")
        response = SearchService._search(query, filter)
        results = SearchService.parse(response)
        return results

    @staticmethod
    def parse(data: Dict[str, Any]):
        results = []
        contents = data["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"]["sectionListRenderer"]["contents"]
        items = contents[0]["itemSectionRenderer"]["contents"] if contents else []
        for item in items:
            if "videoRenderer" in item:
                renderer = item["videoRenderer"]
                results.append(
                    {
                        "video_id": renderer["videoId"],
                        "thumbnail_url": renderer["thumbnail"]["thumbnails"][-1]["url"],
                        "title": "".join(run["text"] for run in renderer["title"]["runs"]),
                    }
                )
        return results

    @staticmethod
    def _search(query: Optional[str] = None, filter: SearchFilter = SearchFilter.VIDEO):
        client = innertube.InnerTube("WEB", "2.20230920.00.00")
        response = client.search(query=query, params=filter.code if filter else None)
        return response

    @staticmethod
    def get_youtube_url(video_id: str) -> str:
        return f"https://www.youtube.com/watch?v={video_id}"

    @staticmethod
    def get_stream(youtube_url):
        try:
            session = streamlink.Streamlink()
            streams = session.streams(youtube_url)
            if streams:
                best_stream = streams.get("best")
                return best_stream.url if best_stream else None
            else:
                gr.Warning(f"No streams found for: {youtube_url}")
                return None
        except Exception as e:
            gr.Error(f"An error occurred while getting stream: {e}")
            logging.warning(f"An error occurred: {e}")
            return None


INITIAL_STREAMS = SearchService.search("world live cams", SearchFilter.LIVE)


class LiveYouTubeObjectDetector:
    def __init__(self):
        logging.getLogger().setLevel(logging.DEBUG)
        self.model = YOLO("yolov8x.pt")
        self.current_page_token = None
        self.streams = INITIAL_STREAMS

        # Gradio UI
        initial_gallery_items = [(stream["thumbnail_url"], stream["title"]) for stream in self.streams]
        self.gallery = gr.Gallery(label="Live YouTube Videos", value=initial_gallery_items, show_label=True, columns=[4], rows=[5], object_fit="contain", height="auto", allow_preview=False)
        self.search_input = gr.Textbox(label="Search Live YouTube Videos")
        self.stream_input = gr.Textbox(label="URL of Live YouTube Video")
        self.annotated_image = gr.AnnotatedImage(show_label=False)
        self.search_button = gr.Button("Search", size="lg")
        self.submit_button = gr.Button("Detect Objects", variant="primary", size="lg")
        self.page_title = gr.HTML("<center><h1><b>Object Detection in Live YouTube Streams</b></h1></center>")

    def detect_objects(self, url):
        stream_url = SearchService.get_stream(url)
        if not stream_url:
            gr.Error(f"Unable to find a stream for: {stream_url}")
            return self.create_black_image(), []
        frame = self.get_frame(stream_url)
        if frame is None:
            gr.Error(f"Unable to capture frame for: {stream_url}")
            return self.create_black_image(), []
        return self.annotate(frame)

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
                logging.warning("Unable to process the HLS stream with cv2.VideoCapture.")
                return None
        except Exception as e:
            logging.warning(f"An error occurred while capturing the frame: {e}")
            return None

    def annotate(self, frame):
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
                # EXTRACT BOUNDING BOX AND CONVERT TO INTEGER
                x1, y1, x2, y2 = box.xyxy[0]
                bbox_coords = (int(x1), int(y1), int(x2), int(y2))
                annotations.append((bbox_coords, class_name))
        return annotations

    def create_black_image():
        black_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        pil_black_image = Image.fromarray(black_image)
        cv2_black_image = cv2.cvtColor(np.array(pil_black_image), cv2.COLOR_RGB2BGR)
        return cv2_black_image

    @staticmethod
    def get_live_streams(query=""):
        return SearchService.search(query if query else "world live cams", SearchFilter.LIVE)

    def render(self):
        with gr.Blocks(title="Object Detection in Live YouTube Streams", css="footer {visibility: hidden}") as app:
            self.page_title.render()
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        self.stream_input.render()
                        self.submit_button.render()
                self.annotated_image.render()
            with gr.Group():
                with gr.Row():
                    self.search_input.render()
                    self.search_button.render()
            with gr.Row():
                self.gallery.render()

            @self.gallery.select(inputs=None, outputs=[self.annotated_image, self.stream_input])
            def detect_objects_from_gallery_item(evt: gr.SelectData):
                if evt.index is not None and evt.index < len(self.streams):
                    selected_stream = self.streams[evt.index]
                    stream_url = SearchService.get_youtube_url(selected_stream["video_id"])
                    frame_output = self.detect_objects(stream_url)
                    return frame_output, stream_url
                return None, ""

            @self.search_button.click(inputs=[self.search_input], outputs=[self.gallery])
            def search_live_streams(query):
                self.streams = self.get_live_streams(query)
                gallery_items = [(stream["thumbnail_url"], stream["title"]) for stream in self.streams]
                return gallery_items

            @self.submit_button.click(inputs=[self.stream_input], outputs=[self.annotated_image])
            def detect_objects_from_url(url):
                return self.detect_objects(url)

        return app.queue().launch(show_api=False, debug=True, quiet=False, share=True)


if __name__ == "__main__":
    LiveYouTubeObjectDetector().render()
