import streamlit as st
from streamlit_cropper import st_cropper
from zipfile import ZipFile
import io
import base64
import uuid
import re
import urllib 
import os
from utils.fileDetail import EXTERNAL_DEPENDENCIES
from PIL import Image

def write_page(page):
	"""Writes the specified page/module
	To take advantage of this function, a multipage app should be structured into sub-files with a `def write()` function
	Arguments:
		page {module} -- A module with a "def write():" function
	"""

	page.write()


def download_button(input_file, button_text):
    """
    Generates a link to download the given object_to_download.

    From: https://discuss.streamlit.io/t/a-download-button-with-custom-css/4220
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    Returns:
    """
    output_file=Image.fromarray(input_file)

    buffered=io.BytesIO()
    output_file.save(buffered, format="PNG")
    img_str=base64.b64encode(buffered.getvalue()).decode()

    button_uuid=str(uuid.uuid4()).replace("-", "")
    button_id=re.sub("\d+", "", button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link=(
          custom_css
          + f'<a download="image.png" id="{button_id}" href="data:file/png;base64,{img_str}">{button_text}</a><br><br>'
    )

    st.markdown(dl_link, unsafe_allow_html=True)


def image_to_buffer(input_image):
    '''
    This function will return the buffer for the input image.
    '''
    
    try: input_image = Image.open(input_image)
    except: print('file cant open by PIL')

    buffered=io.BytesIO()
    input_image.save(buffered, format='PNG')
    return buffered


def image_cropping(input_image, color='#F63366', ratio=(1,1)):
    '''
    This function will show streamlit cropper and result the cropped image
    input_image: the image to be cropped
    color: the color of the cropped box (default: black)
    ratio: the cropped image ratio (default: (1,1))
    '''
    input_image = Image.open(input_image)
    input_image = st_cropper(input_image, realtime_update=True, box_color=color, aspect_ratio=ratio)
    input_image.thumbnail((300,300))
    return input_image


def open_gif(path, display_size=256, col=None):
    '''
    This function will show gif in streamlit.
    path: the path of gif
    display_size: the size to be display (default: 256)
    col: if you are using streamlit beta_column, pass through this parameter
    '''

    file_ = open(path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    if col is None:
        st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="gif" width={display_size}>',
        unsafe_allow_html=True,
        )
    else:
        col.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="gif" width={display_size}>',
        unsafe_allow_html=True,
        )


def download_file(file):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(EXTERNAL_DEPENDENCIES[file]["path"]+file): return
    
    #st.write(EXTERNAL_DEPENDENCIES[file]["path"]+file)
    
    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file)
        progress_bar = st.progress(0)
        with open(EXTERNAL_DEPENDENCIES[file]["path"]+file, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()



