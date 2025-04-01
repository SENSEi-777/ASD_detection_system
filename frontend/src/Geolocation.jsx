import React, { useEffect } from 'react';
import { useGeolocation } from 'react-use';

const Geolocation = ({ setLocation }) => {
  const state = useGeolocation();

  // Handle successful geolocation
  useEffect(() => {
    if (state.latitude && state.longitude) {
      window.parent.postMessage({
        type: 'streamlit:setComponentValue',
        value: {
          lat: state.latitude,
          lng: state.longitude
        }
      }, '*');
    }
  }, [state]);

  // Handle permission denial
  useEffect(() => {
    if (state.error && state.error.code === state.error.PERMISSION_DENIED) {
      alert("Please enable location permissions in your browser settings to use this feature.");
    }
  }, [state.error]);

  return null;
};

export default Geolocation;
