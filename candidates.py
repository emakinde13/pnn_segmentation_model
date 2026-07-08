class Candidate:
    def __init__(self, x, y, radius, confidence, image_id, roi=None, mask=None, accepted=None,):
        self.x = x
        self.y = y
        self.radius = radius
        self.confidence = confidence
        self.image_id = image_id
        self.roi = roi
        self.mask = mask
        self.accepted = accepted
        self.features = {}
