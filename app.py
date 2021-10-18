import streamlit as st

import pages.higan_page as higan_page
import pages.itfgan_page as itfgan_page
import pages.home as home

from utils.helper import write_page

st.set_page_config(
    page_title="GenForce",
    page_icon=":dizzy:",
    layout="wide",
    initial_sidebar_state="expanded",
)


PAGES = {
    'Home': home,
    'InterFaceGAN': itfgan_page,
    'HiGAN': higan_page
}


def main():

    with st.sidebar:
        # update message
        st.info("""
            ⭐ **Update**: New version for Streamlit Cloud.
            """)

        # navigation
        st.title('Navigation')
        page_selection = st.radio("Go to", list(PAGES.keys()))

    page = PAGES[page_selection]
    write_page(page)


if __name__ == "__main__":
    main()
