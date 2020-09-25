import cv2
from imutils.perspective import four_point_transform
from imutils import contours, grab_contours, resize
from numpy import array, concatenate
import yaml
from .exceptions import DataException, NoImageException


# Margin percentage, 50 is the highest number expecting it each bit in the image
# to be exact.  This is used for debug printing, otherwise has no function.
MARGIN_PERC = 40
# Gap between digits of percent of size of digits.
DIGIT_GAP_PERC = 0.2

# Digit or string lookup.  Added characters as well in case the LCD displays strings.
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): '0',
    (0, 0, 1, 0, 0, 1, 0): '1',
    (1, 0, 1, 1, 1, 0, 1): '2',
    (1, 0, 1, 1, 0, 1, 1): '3',
    (0, 1, 1, 1, 0, 1, 0): '4',
    (1, 1, 0, 1, 0, 1, 1): '5',
    (1, 1, 0, 1, 1, 1, 1): '6',
    (0, 1, 0, 1, 1, 1, 1): '6',
    (1, 0, 1, 0, 0, 1, 0): '7',
    (1, 1, 1, 1, 1, 1, 1): '8',
    (1, 1, 1, 1, 0, 1, 1): '9',
    (1, 1, 1, 1, 1, 1, 0): 'A',
    (1, 1, 0, 0, 1, 0, 1): 'C',
    (1, 1, 0, 1, 1, 0, 1): 'E',
    (1, 1, 0, 1, 1, 0, 0): 'F',
    (0, 1, 1, 1, 1, 1, 0): 'H',
    (0, 0, 1, 0, 1, 1, 1): 'J',
    (0, 1, 0, 0, 1, 0, 1): 'L',
    (1, 1, 1, 1, 1, 0, 0): 'P',
    (0, 1, 1, 0, 1, 1, 1): 'U',
}

class RecognizeDigitsHelper:
    def crop(self, image):
        pts = self.options.get('crop')
        if not pts:
            return image

        (x1, x2) = pts.get('x', (0, len(image[1])))
        (y1, y2) = pts.get('y', (0, len(image[0])))
        # Cropping a numpy has x and y backwards from expectation
        return image[y1:y2, x1:x2]

    def warp(self, image):
        pts = self.options.get('warp')
        if not pts:
            return image

        return four_point_transform(image, array(pts))


class RecognizeDigits(RecognizeDigitsHelper):
    """Recognize a digit from a given image

    :param debug: Debug flag to show images and bounding boxes to illustrate how the digit is
        calculated
    :param num_digits: Number of digits in the display, if not provided will be calculated.
    :param digit_gap: Number of pixels between digits, if not provided will be calculated.
    :param segment_width: The multiplification factor percentage of the width of a segment.
        Defaults to 0.25
    :param segment_height: Percent height of segment.  Defaults 0.15
    :param segment_height_center: Percent height of center segment.  Defaults 0.10
    :param has_decimal: Attempts to detect a decimal point.  Defaults to True.
    :param bbox_size: Bounding box size.  Tuple of (w, h) in pixels.
    :param bbox_location: Bounding box location.  Tuple of (x, y) coordinates in pixels.
    :param booleans: list of dictionary defining booleans.  Use for recognition of light being on.
    """
    def __init__(self, image, **options):
        self.options = options
        self._orig = image
        self._debug = options.get('debug', False)

        self.num_digits = self.options.get('num_digits', 0)
        self.worst_margin = 100
        self.valid = True
        self._attrs = {}
        self._color = self.warp(self.crop(image))

    @property
    def name(self):
        return self.options.get('name')

    def gray_image(self):
        """
        Grays the image
        :returns: grayed image
        """
        return cv2.cvtColor(self._orig, cv2.COLOR_BGR2GRAY)

    def digit_cnts(self, cnts):
        """
        Gets the digit countours based on the size of the countour.
        
        :returns: list of countours
        """
        digitCnts = []

        for cnt in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(cnt)

            # if the contour is sufficiently large, it must be a digit
            if w >= 10 and h >= 15:
                digitCnts.append(cnt)

        return digitCnts

    def segments_as_asc(self, seg_on):
        """
        Debug print segments that are on that look like the LCD display.
        """
        if seg_on[0]:
            print(' _ ')
        else:
            print('   ')
        print('{}{}{}'.format(
            '|' if seg_on[1] else ' ',
            '_' if seg_on[3] else ' ',
            '|' if seg_on[2] else ' ',
        ))
        print('{}{}{}'.format(
            '|' if seg_on[4] else ' ',
            '_' if seg_on[6] else ' ',
            '|' if seg_on[5] else ' ',
        ))

    def test_segment(self, coords, roi, i, num):
        """
        Tests if a segment is on based on the segment area if more pixels are on or off.

        :param coords: coordinates of segment.  Tuple of 2 points where each point is a tuple
            of (x, y).
        :param roi: threshold array of the area we are looking at
        """
        ((x1, y1), (x2, y2)) = coords
        # extract the segment ROI, count the total number of
        # thresholded pixels in the segment, and then compute
        # the area of the segment
        segROI = roi[y1:y2, x1:x2]
        total = cv2.countNonZero(segROI)
        area = (x2 - x1) * (y2 - y1)
        if not area:
            return 0

        # Margin is used to determine how well the score is.  How close to 50% off/on pixels we
        # are.  
        perc = int(total / float(area) * 100)
        margin_exceeded = perc > (50 - MARGIN_PERC) and perc < (MARGIN_PERC + 50)
        if self._debug and margin_exceeded:
            print('Digit #{}: Marginal segment {} at {}% {} {}'.format(num, i, perc, total, area))
        # record worst margin for all segments to present a score of how well the reading was.
        abs_margin = abs(perc - 50) * 2
        if abs_margin < self.worst_margin:
            self.worst_margin = abs_margin

        # if the total number of non-zero pixels is greater than
        # 50% of the area, mark the segment as "seg_on"
        if perc > 50:
            return 1
        return 0

    def test_boolean(self, thresh, bbox):
        """
        Tests for a boolean, i.e. a light is on or off.

        :param thresh: threshold array
        :param bbox: Bounding box of boolean, tuple of (x, y, w, h)
        """
        # extract the digit ROI
        (x, y, w, h) = bbox
        roi = thresh[y:y + h, x:x + w]
        if self._debug: cv2.rectangle(self._orig, (x, y), (x+w, y+h), (255,0,0), 1)
        return bool(self.test_segment(((0, 0), (w, h)), roi, 10, 0))

    def read_digit(self, thresh, bbox, num):
        """
        Tests for a boolean, i.e. a light is on or off.

        :param thresh: threshold array
        :param bbox: Bounding box of boolean, tuple of (x, y, w, h)
        :param num: Digit number
        :returns: digit as a string, may contain a decimal point in addition
        """
        (x, y, tw, h) = bbox
        # calculate this digits width with the digit gap
        w = int(round(tw / self.num_digits))
        # calculate x offset for our digit
        x = x + num * w
        # Use the provided digit gap or calculate it
        digit_gap = self.options.get('digit_gap', int(round(w * DIGIT_GAP_PERC)))
        # Offset the width and x coord by the digit gap
        w -= digit_gap
        x += int(round(digit_gap * num * DIGIT_GAP_PERC))

        roi = thresh[y:y + h, x:x + w]

        # compute the width and height of each of the 7 segments
        # we are going to examine
        segment_width = self.options.get('segment_width', 0.25)
        segment_height = self.options.get('segment_height', 0.15)
        segment_height_center = self.options.get('segment_height_center', 0.1)
        (roi_h, roi_w) = roi.shape
        (dw, dh) = (int(roi_w * segment_width), int(roi_h * segment_height))
        dhc = int(roi_h * segment_height_center)

        # define the set of 7 segments
        segments = [
            ((dw, 0), (w - dw, dh)),    # top
            ((0, dh), (dw, h // 2 - dhc)),    # top-left
            ((w - dw, dh), (w, h // 2 - dhc)),    # top-right
            ((dw, (h // 2) - dhc) , (w - dw, (h // 2) + dhc)), # center
            ((0, h // 2 + dhc), (dw, h - dh)),    # bottom-left
            ((w - dw, h // 2 + dhc), (w, h - dh)),    # bottom-right
            ((dw, h - dh), (w - dw, h))    # bottom
        ]

        if self._debug: cv2.rectangle(self._color, (x, y), (x+w, y+h), (255,255,0), 1)

        seg_on = [0] * len(segments)
        # loop over the segments
        for (i, coords) in enumerate(segments):
            seg_on[i] = self.test_segment(coords, roi, i, num)

            # Show each segment block as a green outline
            if self._debug:
                ((x1, y1), (x2, y2)) = coords
                cv2.rectangle(self._color, (x+x1, y+y1), (x+x2, y+y2), (0,255,0), 1)

        # Lookup the digit, if we cannot identify return unknown and print it out.
        digit = DIGITS_LOOKUP.get(tuple(seg_on), 'unknown')
        if digit == 'unknown':
            self.segments_as_asc(seg_on)

        # Check for a decimal point and append it to the digit string if its on
        if self.options.get('has_decimal', True):
            decimal_coords = ((w + 1, h - dh), (w + dw, h))
            ((x1, y1), (x2, y2)) = decimal_coords
            roi = thresh[y:y + h, x:x + w + dw]
            cv2.rectangle(self._color, (x+x1, y+y1), (x+x2, y+y2), (0,255,0), 1)
            if self.test_segment(decimal_coords, roi, 7, num):
                return [digit, '.']
        return [digit]

    def read(self):
        """
        Reads the digital display.

        :returns: string of the digits read.
        """
        gray = self.gray_image()
        cropped = self.crop(gray)
        if self._debug: cv2.imshow("cropped", cropped)
        warped = self.warp(cropped)
        if self._debug: cv2.imshow("warped", warped)

        # threshold the warped image
        thresh = cv2.threshold(warped, 127, 255, cv2.THRESH_OTSU)[1]

        # find contours in the thresholded image, then initialize the
        # digit contours lists
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = grab_contours(cnts)

        dcnts = self.digit_cnts(cnts)
        if not dcnts:
            # Show the user what we are dealing with when debug is on when we cannot
            # recognize any digits.
            if self._debug:
                cv2.imshow('Attributes', self._orig)
                cv2.imshow("orig", self._color)
                cv2.waitKey(0)
            self.valid = False
            raise DataException("Could not identify digits.")

        # sort only useful when using the contours to define the bounding box for each digit
        digitCnts = contours.sort_contours(dcnts, method="left-to-right")[0]
        if not self.num_digits:
            self.num_digits = len(digitCnts)

        # Get calculated bounding box for all digits in display
        x, y, w, h = cv2.boundingRect(concatenate(digitCnts))

        # override calculated value with user given ones if provided
        w, h = self.options.get('bbox_size', (w, h))
        x, y = self.options.get('bbox_location', (x, y))

        if self._debug:
            # draw rectangle of bounding box
            self._thresh = thresh.copy()
            cv2.rectangle(self._color, (x, y), (x+w, y+h), (255,0,0), 1)

        # Read each digit and place result in a string
        digits = []
        for num in range(self.num_digits):
            digits.extend(self.read_digit(thresh, (x, y, w, h), num))

        # Test the booleans
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU)[1]
        for name, coords in self.options.get('booleans', {}).items():
            self._attrs[name] = self.test_boolean(thresh, coords)

        if self._debug:
            cv2.imshow('Attributes', self._orig)
            cv2.imshow("thresh", self._thresh)
            cv2.imshow("orig", self._color)
            cv2.imwrite(f"{self.options.get('name')}_bounding_box.png", self._color)
            cv2.waitKey(0)

        return "".join(digits)

    @property
    def as_int(self):
        """
        Read as an integer.

        :returns: integer of digits read
        """
        try:
            return int(self.read())
        except ValueError:
            return ''

    @property
    def as_float(self):
        """
        Read as a float.

        :returns: float of digits read
        """
        try:
            return float(self.read())
        except ValueError:
            return ''

    @property
    def attributes(self):
        """
        Contain boolean attributes input, in addition to margin score and if the data is valid.

        :returns: float of digits read
        """
        data = self._attrs
        data["margin"] =  self.worst_margin
        data["valid"] =  self.valid
        return data



class ReadDigitsImage(RecognizeDigitsHelper):
    def __init__(self, yaml_path, debug=False):
        self.debug = debug
        self.options = self.read_yaml(yaml_path)

    def read_yaml(self, yaml_path):
        """
        Reads the yaml file.
        """
        with open(str(yaml_path)) as yfile:
            data = yaml.load(yfile, Loader=yaml.FullLoader)
        return data

    def capture(self):
        """
        Capture the image using the credentials provided within the yaml options.

        :returns: image
        :raise: NoImageException
        """
        cap = cv2.VideoCapture('rtsp://{}:{}@{}'.format(
            self.options['username'],
            self.options['password'],
            self.options['ip'],
        ))
        ret, image = cap.read()
        if image is None:
            raise NoImageException('Could not capture image.')

        return image

    def conditioned_image(self):
        """
        Pre condition the image by cropping and warping it.  Finally resize to set size.
        This resize helps when the resulting crop image is small to get better accuracy.

        :returns: image
        """
        image = self.capture()

        # Raw crop to get just the device we are looking for
        cropped = self.crop(image)

        # Warp it back to shape, do this by framing the device with lines
        warped = self.warp(cropped)

        # Resize to larger for more accuracy.
        return resize(warped, height=500)

    def digits(self):
        """
        Get digits from all the images specified in a dictionary.

        :returns: dict of RecognizeDigits
        """
        image = self.conditioned_image()
        recd = {}
        for dgt in self.options['digits']:
            recd[dgt['name']] = RecognizeDigits(image, debug=self.debug, **dgt)

        return recd

    def save(self, path):
        """
        Save the result as json to the file path specified.

        :returns: True if successful, False if the digits could not be read.
        """
        recd = self.digits()
        if not recd:
            return False

        outdict = {name: dgt.as_dict for name, dgt in recd}
        with open(path, 'w') as outfile:
            json.dump(recd, outfile)

        return True

    def debug_images(self):
        """
        Helper function to display images for cropping and warping for adjusting input parameters.
        """
        image = self.capture()

        # Raw crop to get just the device we are looking for
        cropped = self.crop(image)

        # Warp it back to shape, do this by framing the device with lines
        warped = self.warp(cropped)

        # Draw the warp to line on the full cropped image
        pts = array(self.options.get('warp'))
        cv2.polylines(cropped,[pts],False,(0,255,0),1)

        # Resize
        resized = resize(warped, height=500)
        cv2.imshow("cropped", cropped)
        cv2.imwrite("cropped.png", cropped)
        cv2.imshow("warped", resized)
        cv2.imwrite("warped.png", resized)

        for dgt in self.options['digits']:
            name = dgt.get('name')
            image = resized.copy()
            recd = RecognizeDigits(image, **dgt)

            cropped = recd.crop(image)
            warped = recd.warp(cropped)

            # Draw the warp to line on the per display cropped image
            pts = array(dgt.get('warp'))
            cv2.polylines(cropped,[pts],False,(0,255,0),2)
            cv2.imshow(name, cropped)
            cv2.imwrite(name + ".png", cropped)

            bbox_location = dgt.get('bbox_location')
            w, h = dgt.get('bbox_size')
            x, y = bbox_location
            # Draw the box around the digit display
            cv2.rectangle(warped, (x, y), (x+w, y+h), (255,0,0), 1)
            cv2.imshow(f"{name} warped", warped)
            cv2.imwrite(f"{name} warped.png", warped)

            # Draw boxes around the booleans and label them
            booleans = resized
            for bname, coords in dgt.get('booleans', {}).items():
                x, y, w, h = coords
                cv2.rectangle(booleans, (x, y), (x+w, y+h), (255,0,0), 1)
                cv2.putText(
                    booleans,
                    "{} {}".format(name, bname),
                    (x, y), 0, 1, (255,0,0), 2
                )

        cv2.imshow(f"{name} booleans", booleans)
        cv2.imwrite(f"{name} booleans.png", booleans)
        cv2.waitKey(0)
