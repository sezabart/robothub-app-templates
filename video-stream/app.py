from depthai_sdk import OakCamera
from robothub_oak import LiveView, BaseApplication
from robothub_oak.data_processors import BaseDataProcessor


class ObjectDetection(BaseDataProcessor):
    """
    This class is a data processor that will process the output from the pipeline.
    In this example we use the NN output to detect objects and send an event to the frontend.
    """

    def __init__(self, live_view: LiveView):
        super().__init__()
        self.live_view = live_view

    def process_packets(self, packet):  # This method has to be implemented
        # Iterate over all detections and add a rectangle to the live view
        for detection in packet.detections:
            bbox = [*detection.top_left, *detection.bottom_right]
            self.live_view.add_rectangle(bbox, label=detection.label)

        # Publish the frame
        self.live_view.publish(packet.frame)


class ExampleApplication(BaseApplication):
    """
    This is an example application that shows how to use the OakCamera class to create an object detection pipeline.
    In this example we use the YOLOv6 model to detect objects from the color camera.
    """

    def __init__(self):
        super().__init__()  # Add any initialization code here

    def setup_pipeline(self, device: OakCamera):  # This method has to be implemented
        """
        This method is the entrypoint for each device.
        OakCamera is a class from the DepthAI SDK package that provides a simple interface to create pipelines.
        Documentation for the DepthAI SDK can be found here: https://docs.luxonis.com/projects/sdk/en/latest/.
        """
        color = device.create_camera(source="color", fps=30, resolution="1080p", encode="mjpeg")
        nn = device.create_nn(model='yolov6nr3_coco_640x352', input=color)

        # Create a live view for the color camera. This will be displayed in the frontend and RobotHub app
        live_view = LiveView.create(
            device=device,
            component=color,
            unique_key="color_stream",  # Unique key is used to identify the live view in the frontend
            name="Object detection",  # Stream name
            manual_publish=True  # We will publish the frame manually in the data processor, disable if you want to stream the raw camera feed
        )

        # Create a data processor that will process the NN output and send an event to the frontend
        object_detection = ObjectDetection(live_view)

        # Create a callback that will be called when the NN has processed a frame
        # BaseDataProcessor implements the __call__ method, so we can pass it as a callback
        device.callback(nn.out.main, object_detection)
