export default function omit(obj, fields) {
  const clone = Object.assign({}, obj);
  if (Array.isArray(fields)) {
    fields.forEach(key => {
      delete clone[key];
    });
  }
  return clone;
}