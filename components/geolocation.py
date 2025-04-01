import streamlit.components.v1 as components
def get_geolocation():
    return components.declare_component(
        "geolocation",
        url="https://asddetection-webapp.streamlit.app/"  # For production
    )
