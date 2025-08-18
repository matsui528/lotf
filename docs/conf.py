# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import tomllib
from importlib.metadata import metadata, PackageNotFoundError

# Add the source directory to the Python path
sys.path.insert(0, os.path.abspath("../src"))


def extract_author_from_metadata(pkg_metadata):
    """Extract author name from package metadata.

    Modern pyproject.toml with authors array gets converted to Author-email field
    in the format "Name <email>" rather than separate Author field.
    """
    author = pkg_metadata.get("Author")
    if author:
        return author

    author_email = pkg_metadata.get("Author-email", "")
    if not author_email:
        return None

    # Extract name from "Name <email>" format
    if "<" in author_email:
        return author_email.split("<")[0].strip()
    else:
        # Fallback: use part before @ as name
        return author_email.split("@")[0]


def get_project_metadata():
    """Get project metadata from installed package or pyproject.toml fallback."""
    try:
        # Prefer installed package metadata
        pkg_metadata = metadata("lotf")
        return {
            "name": pkg_metadata["Name"],
            "version": pkg_metadata["Version"],
            "author": extract_author_from_metadata(pkg_metadata),
        }
    except PackageNotFoundError:
        # Fallback to pyproject.toml for development
        with open("../pyproject.toml", "rb") as f:
            pyproject_data = tomllib.load(f)

        project_info = pyproject_data["project"]
        return {
            "name": project_info["name"],
            "version": project_info["version"],
            "author": project_info["authors"][0]["name"],
        }


# Get project metadata
metadata_info = get_project_metadata()
release = metadata_info["version"]
version = release  # The short X.Y version

# -- Project information -----------------------------------------------------
project = f"{metadata_info['name']} documentation"
copyright = f"2025, {metadata_info['author']}"
author = metadata_info["author"]

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "lotf documentation"
html_short_title = "lotf documentation"

html_theme_options = {
    "sidebar_hide_name": False,  # Show the project name in sidebar
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/matsui528/lotf/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#2563eb",
    },
    "dark_css_variables": {
        "color-brand-primary": "#3b82f6",
        "color-brand-content": "#3b82f6",
    },
}

html_static_path = ["_static"]

# -- Options for autodoc ----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    # "faiss": ("https://faiss.ai/cpp_api/", None),  # URL not available, disabled for now
}

# -- Options for copybutton -------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
