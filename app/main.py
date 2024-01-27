"""
Runs the streamlit app. 

Call this file in the terminal (from the `traingenerator` dir) 
via `streamlit run app/main.py`.
"""

import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from github import Github
from jinja2 import Environment, FileSystemLoader

import utils

MAGE_EMOJI_URL = "https://symbl-world.akamaized.net/i/webp/5f/643cfb26a7a37a8d1aef71619b0d10.webp"
st.set_page_config(
    page_title="Traingenerator", page_icon=MAGE_EMOJI_URL,
)

# Set up github access for "Open in Colab" button.
# TODO: Maybe refactor this to another file.
load_dotenv()  # load environment variables from .env file
if os.getenv("GITHUB_TOKEN") and os.getenv("REPO_NAME"):
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo(os.getenv("REPO_NAME"))
    colab_enabled = True


    def add_to_colab(notebook):
        """Adds notebook to Colab by pushing it to Github repo and returning Colab link."""
        notebook_id = str(uuid.uuid4())
        repo.create_file(
            f"notebooks/{notebook_id}/generated-notebook.ipynb",
            f"Added notebook {notebook_id}",
            notebook,
        )
        colab_link = f"http://colab.research.google.com/github/{os.getenv('REPO_NAME')}/blob/main/notebooks/{notebook_id}/generated-notebook.ipynb"
        return colab_link


else:
    colab_enabled = False

# Display header.
st.markdown("<br>", unsafe_allow_html=True)
st.image(MAGE_EMOJI_URL, width=80)

"""
# LiDCLS-Species-Classification
### Code Generator for Machine Learning
"""
st.markdown("<br>", unsafe_allow_html=True)
"""Jumpstart your machine learning code:

1. Specify model in the sidebar *(click on **>** if closed)*
2. Training code will be generated below
3. Download and do magic! :sparkles:

---
"""

task = 'Image classification'
framework = 'PyTorch'
template_dir = 'templates/Image classification_PyTorch'

# Show selectors for task and framework in sidebar (based on template_dict). These
# selectors determine which template (from template_dict) is used (and also which
# template-specific sidebar components are shown below).
with st.sidebar:
    st.info(
        "🎈 **NEW:** Train and Test with your own model !"
    )

# Show template-specific sidebar components (based on sidebar.py in the template dir).
template_sidebar = utils.import_from_file(
    "template_sidebar", os.path.join(template_dir, "sidebar.py")
)
inputs = template_sidebar.show()

# Generate code and notebook based on template.py.jinja file in the template dir.
env = Environment(
    loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True,
)
template = env.get_template("code-template.py.jinja")
code = template.render(header=utils.code_header, notebook=False, **inputs)
notebook_code = template.render(header=utils.notebook_header, notebook=True, **inputs)
notebook = utils.to_notebook(notebook_code)

# Display donwload/open buttons.
# TODO: Maybe refactor this (with some of the stuff in utils.py) to buttons.py.
st.write("")  # add vertical space
col1, col2, col3 = st.columns(3)
open_colab = col1.button("🚀 Open in Colab")  # logic handled further down
with col2:
    utils.download_button(code, "generated-code.py", "🐍 Download (.py)")
with col3:
    utils.download_button(notebook, "generated-notebook.ipynb", "📓 Download (.ipynb)")
colab_error = st.empty()

# Display code.
# TODO: Think about writing Installs on extra line here.
st.code(code)

# Handle "Open Colab" button. Down here because to open the new web page, it
# needs to create a temporary element, which we don't want to show above.
if open_colab:
    if colab_enabled:
        colab_link = add_to_colab(notebook)
        utils.open_link(colab_link)
    else:
        colab_error.error(
            """
            **Colab support is disabled.** (If you are hosting this: Create a Github 
            repo to store notebooks and register it via a .env file)
            """
        )
