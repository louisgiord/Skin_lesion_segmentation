import skimage.morphology as morph

struct_element1 = morph.disk(3)

def dilate(img, struct_element):
    return morph.dilation(img, struct_element)

struct_element2 = morph.rectangle(2, 5)

def erode(img, struct_element):
    return morph.erosion(img, struct_element)

def overall(img, struct_element1, struct_element2):
    return dilate(erode(img, struct_element2), struct_element1)



