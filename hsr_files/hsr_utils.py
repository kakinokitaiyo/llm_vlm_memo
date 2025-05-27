from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel, StereoCameraModel


class HSRUtil():
    def __init__(self, ri):
        self.ri = ri
        self.bridge = CvBridge()

    def getHeadRGBDImages(self, timeout=None):
        topic_names = ['/hsrb/head_rgbd_sensor/rgb/image_rect_color',
                       '/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw',
                       '/hsrb/head_rgbd_sensor/depth_registered/camera_info']
        topic_types = [Image, Image, CameraInfo]
        sync_sub = self.ri.oneShotSyncSubscriber(topic_names, topic_types)
        sync_sub.waitResults(timeout=timeout)
        sub_data = sync_sub.data()
        if sub_data is not None:
            color_image = self.bridge.imgmsg_to_cv2(
                sub_data[0], desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(
                sub_data[1], desired_encoding='passthrough')
            camera_model = PinholeCameraModel()
            camera_model.fromCameraInfo(sub_data[2])
            ret_dat = (color_image, depth_image, camera_model)
        else:
            ret_dat = None, None, None
        return ret_dat

    def getHandCameraIamge(self, timeout=None):
        topic_names = ['/hsrb/hand_camera/image_raw',
                       '/hsrb/hand_camera/camera_info']
        topic_types = [Image, CameraInfo]
        sync_sub = self.ri.oneShotSyncSubscriber(topic_names, topic_types)
        sync_sub.waitResults(timeout=timeout)
        sub_data = sync_sub.data()
        if sub_data is not None:
            color_image = self.bridge.imgmsg_to_cv2(
                sub_data[0], desired_encoding='bgr8')
            camera_model = PinholeCameraModel()
            camera_model.fromCameraInfo(sub_data[1])
            ret_dat = (color_image,  camera_model)
        else:
            ret_dat = None, None
        return ret_dat

    def getStereoCameraImage(self, timeout=None):
        topic_names = ['/hsrb/head_l_stereo_camera/image_rect_color',
                       '/hsrb/head_l_stereo_camera/camera_info',
                       '/hsrb/head_r_stereo_camera/image_rect_color',
                       '/hsrb/head_r_stereo_camera/camera_info']
        topic_types = [Image, CameraInfo, Image, CameraInfo]
        sync_sub = self.ri.oneShotSyncSubscriber(topic_names, topic_types)
        sync_sub.waitResults(timeout=timeout)
        sub_data = sync_sub.data()
        if sub_data is not None:
            left_color_image = self.bridge.imgmsg_to_cv2(
                sub_data[0], desired_encoding='bgr8')
            right_color_image = self.bridge.imgmsg_to_cv2(
                sub_data[2], desired_encoding='bgr8')
            camera_model = StereoCameraModel()
            camera_model.fromCameraInfo(sub_data[1], sub_data[3])
            ret_dat = (left_color_image, right_color_image, camera_model)
        else:
            ret_dat = None, None, None
        return ret_dat
    
    def getCenterCameraImage(self, timeout=None):
        topic_names = ['/hsrb/head_center_camera/image_raw',
                       '/hsrb/head_center_camera/camera_info']
        topic_types = [Image, CameraInfo]
        sync_sub = self.ri.oneShotSyncSubscriber(topic_names, topic_types)
        sync_sub.waitResults(timeout=timeout)
        sub_data = sync_sub.data()
        if sub_data is not None:
            color_image = self.bridge.imgmsg_to_cv2(
                sub_data[0], desired_encoding='bgr8')
            camera_model = PinholeCameraModel()
            camera_model.fromCameraInfo(sub_data[1])
            ret_dat = (color_image,  camera_model)
        else:
            ret_dat = None, None
        return ret_dat


if __name__ == '__main__':
    exec(open('/choreonoid_ws/install/share/irsl_choreonoid/sample/irsl_import.py').read())
    ri = RobotInterface('robotinterface.yaml')
    hsr_util = HSRUtil(ri)
    dats = hsr_util.getHeadRGBDImages()
    dats = hsr_util.getHandCameraIamge()
    dats = hsr_util.getStereoCameraImage()
    dats = hsr_util.getCenterCameraImage()