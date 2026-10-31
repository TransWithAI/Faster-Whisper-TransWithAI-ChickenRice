import { useState } from 'react';
import isMobile from "../isMobile";
import useLayoutEffect from "./useLayoutEffect";

/**
 * Hook to detect if the user is on a mobile device
 * Notice that this hook will only detect the device type in effect, so it will always be false in server side
 */
const useMobile = () => {
  const [mobile, setMobile] = useState(false);
  useLayoutEffect(() => {
    setMobile(isMobile());
  }, []);
  return mobile;
};
export default useMobile;