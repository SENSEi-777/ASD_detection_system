import React, { useEffect } from 'react';
import { useGeolocation } from 'react-use';

const Geolocation = ({ setLocation }) => {
  const state = useGeolocation();

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

  return null;
};

export default Geolocation;
