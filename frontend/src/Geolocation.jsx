import React, { useEffect } from 'react';
import { useGeolocation } from 'react-use';

const Geolocation = () => {
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
    
    if (state.error) {
      window.parent.postMessage({
        type: 'streamlit:setComponentValue',
        value: null
      }, '*');
    }
  }, [state]);

  return (
    <div style={{ display: 'none' }}>
      Geolocation Component - Hidden from View
    </div>
  );
};

export default Geolocation;
