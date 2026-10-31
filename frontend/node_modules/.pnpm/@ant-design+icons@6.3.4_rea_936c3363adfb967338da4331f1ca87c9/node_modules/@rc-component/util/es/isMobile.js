import isMobile from 'is-mobile';
let cached;
const getIsMobile = () => {
  if (typeof cached === 'undefined') {
    cached = isMobile();
  }
  return cached;
};
export default getIsMobile;