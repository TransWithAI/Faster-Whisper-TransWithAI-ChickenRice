import isFragment from "../React/isFragment";
import React from 'react';
export default function toArray(children, option = {}) {
  let ret = [];
  React.Children.forEach(children, child => {
    if ((child === undefined || child === null) && !option.keepEmpty) {
      return;
    }
    if (Array.isArray(child)) {
      ret = ret.concat(toArray(child));
    } else if (isFragment(child) && child.props) {
      ret = ret.concat(toArray(child.props.children, option));
    } else {
      ret.push(child);
    }
  });
  return ret;
}