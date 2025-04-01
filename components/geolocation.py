import streamlit.components.v1 as components

def get_geolocation():
    return components.declare_component(
        "geolocation",
        path="./frontend/build"  # Point to built files
    )
