import { Tensor, Scalar, Rank, stack, scalar, maximum, tensor, norm, abs, square, squaredDifference, atan, sin, cos, print, minimum} from "@tensorflow/tfjs";
import { canvasSize } from "./Canvas";
import * as _ from "lodash";
import { min } from 'lodash';

export const objDict = {
  equal: (x: Tensor, y: Tensor) => squaredDifference(x, y),

  above: ([t1, top]: [string, any], [t2, bottom]: [string, any], offset = 100) =>
    // (getY top - getY bottom - offset) ^ 2
    square(top.y.contents.sub(bottom.y.contents).sub(scalar(offset))),

  sameCenter: ([t1, s1]: [string, any], [t2, s2]: [string, any]) => {
    return distsq(center(s1), center(s2));
  },

  equalLength: ([t1, s1]: [string, any], [t2, s2]: [string, any]): Tensor => {
    if (linelike([t1, t2])) {
      const lpts1 = linePtsT(s1);
      const lpts2 = linePtsT(s2);
      return squaredDifference(dist(lpts1[0], lpts1[1]), dist(lpts2[0], lpts2[1])).mul(100.0);
    } else throw new Error(`${[t1, t2]} not supported for equalLength`)
  },


  below: ([t1, bottom]: [string, any], [t2, top]: [string, any], offset = 100) => 
    square(top.y.contents.sub(bottom.y.contents).sub(scalar(offset))),
    // can this be made more efficient (code-wise) by calling "above" and swapping arguments? - stella


  centerLabel: ([t1, arr]: [string, any], [t2, text1]: [string, any], w = 1.0): Tensor => {
      // console.log("shapes:", "arr pathData", arr.pathData.contents); 

    if (linelike([t1]) && t2 === "Text") {
      const startPt = stack([arr.startX.contents, arr.startY.contents]);
      const endPt = stack([arr.endX.contents, arr.endY.contents]);
      const m = slope(arr.startX.contents, arr.startY.contents, arr.endX.contents, arr.endY.contents);
      // TODO - officially decide on the best angle offset (pi / ?)
      const newAngle  = atan(m).add(scalar(Math.PI / 25).mul(w)); // add pi/15 degrees to angle of arrow 
      const xadd = cos(newAngle).mul(dist(startPt, endPt)); // change in x from start pt is cos(theta) * hypotenuse
      const yadd = sin(newAngle).mul(dist(startPt, endPt));
      let mx;
      let my;

      // not exactly sure why the 2 cases are necessary but they are
      if (arr.startX.contents.greater(arr.endX.contents).dataSync()[0]) {
        mx = arr.endX.contents.add(xadd.div(scalar(2.0))); // move along the x axis by half of what the change in x to get the end pt would be - i.e. find new midpoint of line
        my = arr.endY.contents.add(yadd.div(scalar(2.0)));
      }
      else {
        mx = arr.startX.contents.add(xadd.div(scalar(2.0)));
        my = arr.startY.contents.add(yadd.div(scalar(2.0)));
      }
      // final equation is (mx - lx) ^ 2 + (my - ly) ^ 2 
      // TODO - test on longer/wider labels
      if (m.greater(zero).dataSync()[0]) {
        // if slope is positive label will be on top left of arrow, otherwise on top right
        // put bottom right corner of bb at midpt
        return squaredDifference(mx, text1.x.contents.add(text1.w.contents.div(scalar(2.0)))).add(squaredDifference(my, text1.y.contents.sub(text1.h.contents.div(scalar(2.0)))));
      }
      else {
        // bottom left corner
        return squaredDifference(mx, text1.x.contents.sub(text1.w.contents.div(scalar(2.0)))).add(squaredDifference(my, text1.y.contents.sub(text1.h.contents.div(scalar(2.0)))));
      }
    } else throw new Error(`${[t1, t2]} not supported for centerLabel`)
  },

  //  finds closest side midpoint
  connectMid: ([t1, s1]: [string, any], [touchx, touchy]: [Tensor, Tensor]): Tensor => {
    if (t1 === "Rectangle") {
      const rsides = sideMidpts(s1);
      const pt = stack([touchx, touchy]);
      let closestv = scalar(9999.0) as Tensor; // initialization
      let closests = tensor([0, 0]);
      rsides.forEach(rside => {
      // loop through and find midpoint closest to point
        if (dist(rside, pt).less(closestv).dataSync()[0]) {
            closestv = dist(rside, pt);
            closests = rside;
        }
      });
      // print(squaredDifference(startarr, closests1).add(squaredDifference(endarr, closests2)));
      return dist(closests, pt).mul(3000.0);
    } 
    else throw new Error(`${[t1]} not supported for connectMid`)
  },

  // encourages line to horizontal or vertical
  cardinal: ([t1, s1]: [string, any]) : Tensor => {
    if (linelike([t1])) {
      const m = slope(s1.startX.contents, s1.startY.contents, s1.endX.contents, s1.endY.contents);
      if (abs(m).less(1.0).dataSync()[0]) {
        // horiz line
        return squaredDifference(s1.startY.contents, s1.endY.contents).mul(50);
      }
      else return squaredDifference(s1.startX.contents, s1.endX.contents).mul(50);  // vertical
    }
    else throw new Error(`${[t1]} not supported for cardinal`); // TODO - factor out manually writing fn names in error messages
  },

  sameLine: ([t1, s1]: [string, any], [t2, s2]: [string, any]) : Tensor => {
    if (linelike([t1, t2])) {
      const m1 = slope(s1.startX.contents, s1.startY.contents, s1.endX.contents, s1.endY.contents);
      const m2 = slope(s1.startX.contents, s1.startY.contents, s2.endX.contents, s2.endY.contents);
      return squaredDifference(m1, m2);
    }
    else throw new Error(`${[t1, t2]} not supported for sameLine`);
  },

  // TODO - MAKE MORE GENERAL
  totalMinLength: ([t1, s1]: [string, any], [t2, s2]: [string, any], [t3, s3]: [string, any]): Tensor => {
    if (linelike([t1, t2, t3])) {
      const arrpts1 = linePtsT(s1);
      const arrpts2 = linePtsT(s2);
      const arrpts3 = linePtsT(s3);
      return dist(arrpts1[0], arrpts1[1]).add(dist(arrpts2[0], arrpts2[1])).add(dist(arrpts3[0], arrpts3[1])).mul(100);  // regular energy is v smol so we boost it
    } else throw new Error(`${[t1, t2, t3]} not supported for totalMinLength`);
  },

  minLength: ([t1, s1]: [string, any]): Tensor => {
    if (typesAre([t1], ["Curve"])) {
      const totald = zero;
      for (let i = 0; i < s1.pathData.length - 1; i++) {
        totald.add(dist(tensor(s1.pathData[i]), s1.pathData[i + 1]));
      }
      return totald;
    } else if (linelike([t1])) {
      const arrpts = linePtsT(s1);
      // return distsq(arrpts[0], arrpts[1]).sub(scalar(20.0));
      return dist(arrpts[0], arrpts[1]).mul(500);  // regular energy is v smol so we boost it
    } else throw new Error(`${[t1]} not supported for minLength`);
  },

  // Generic repel function for two GPIs with centers
  repel: ([t1, s1]: [string, any], [t2, s2]: [string, any]) => {
    // HACK: `repel` typically needs to have a weight multiplied since its magnitude is small
    // TODO: find this out programmatically
    const repelWeight = 10e6;
    if (t1 === "Rectangle" && linelike([t2])) {   // TODO - NEEDS TESTING
      const lpts = linePtsT(s2);
      const cp = closestptPtSeg(center(s1), lpts[0], lpts[1]);
      return distsq(center(s1), cp).add(epsd).reciprocal().mul(repelWeight);
    }
    // 1 / (d^2(cx, cy) + eps)
    else return distsq(center(s1), center(s2)).add(epsd).reciprocal().mul(repelWeight);
  },

  centerArrow: ([t1, arr]: [string, any], [t2, text1]: [string, any], [t3, text2]: [string, any]): Tensor => {
    const spacing = scalar(1.1); // arbitrary

    if (typesAre([t1, t2, t3], ["Arrow", "Text", "Text"])) {
      // HACK: Arbitrarily pick the height of the text
      // [spacing * getNum text1 "h", negate $ 2 * spacing * getNum text2 "h"]
      return centerArrow2(arr, center(text1), center(text2),
        [spacing.mul(text1.h.contents),
        text2.h.contents.mul(spacing).mul(scalar(1.0)).neg()]);
    } else throw new Error(`${[t1, t2, t3]} not supported for centerArrow`);
  },

};

export const constrDict = {
  maxSize: ([shapeType, props]: [string, any]) => {
    const limit = scalar(Math.max(...canvasSize));
    switch (shapeType) {
      case "Circle":
        return stack([props.r.contents, limit.div(scalar(6.0)).neg()]).sum();
      case "Square":
        return stack([props.side.contents, limit.div(scalar(3.0)).neg()]).sum();
      default:
        // HACK: report errors systematically
        throw new Error(`${shapeType} doesn't have a maxSize`);
    }
  },

  minSize: ([shapeType, props]: [string, any]) => {
    const limit = scalar(20);
    switch (shapeType) {
      case "Circle":
        return stack([limit, props.r.contents.neg()]).sum();
      case "Square":
        return stack([limit, props.side.contents.neg()]).sum();
      default:
        // HACK: report errors systematically
        throw new Error(`${shapeType} doesn't have a minSize`);
    }
  },

  connect: ([t1, s1]: [string, any], [touchx, touchy]: [Tensor, Tensor]): Tensor => {
    if (t1 === "Rectangle") {
      return touchPerim(s1, touchx, touchy)
    } 
    else if (t1 === "Circle") {
      return dist(stack([touchx, touchy]), center(s1))
    }
    else throw new Error(`${[t1]} not supported for connect`)
  },
   
  contains: (
    [t1, s1]: [string, any],
    [t2, s2]: [string, any],
    offset: Tensor
  ) => {
    if (t1 === "Circle" && t2 === "Circle") {
      const d = dist(center(s1), center(s2));
      // const o = s1.r.contents.sub(s2.r.contents);
      const o = offset
        ? s1.r.contents.sub(s2.r.contents).sub(offset)
        : s1.r.contents.sub(s2.r.contents);
      return d.sub(o);
    } else if (t1 === "Circle" && t2 === "Text") {
      const d = dist(center(s1), center(s2));
      const textR = maximum(s2.w.contents, s2.h.contents);
      return d.sub(s1.r.contents).add(textR);
    } else if (t1 === "Square" && t2 === "Circle"){
      // dist (outerx, outery) (innerx, innery) - (0.5 * outer.side - inner.radius)
      const sq = stack([s1.x.contents, s1.y.contents]);
      const d = dist(sq, center(s2));
      return d.sub(scalar(0.5).mul(s1.side.contents).sub(s2.r.contents));
    } else if (t1 === "Rectangle" && t2 === "Text") {
      // dist (getX l, getY l) (getX r, getY r) - getNum r "w" / 2 +
      // getNum l "w" / 2 + padding
      const d = dist(center(s1), center(s2));
      return d.sub(s1.w.contents.div(scalar(2.0))).add(s2.w.contents.div(scalar(2.0))).add(offset);
     } else {
      console.error(`${[t1, t2]} not supported for contains`);
      return scalar(0.0);

      // TODO revert
      // throw new Error(`${[t1, t2]} not supported for contains`);
    }
  },

  // TODO  - allow for user reordering of parameters
  disjoint: ([t1, s1]: [string, any], [t2, s2]: [string, any]) => {
    if (t1 === "Circle" && t2 === "Circle") {
      const d = dist(center(s1), center(s2));
      const o = stack([s1.r.contents, s2.r.contents, 10]);
      return o.sum().sub(d);
    }
    else if (t1 === "Circle" && t2 === "Rectangle") {
      const d = dist(center(s1), center(s2));
      const o = stack([s1.r.contents, s2.w.contents.mul(scalar(0.5)), 10]); // should be made more exact
      return o.sum().sub(d);
    }
    else if (t1 === "Rectangle" && (t2 ==="Rectangle" || t2 === "Text")) {
      const d = dist(center(s1), center(s2));
      const o = stack([s1.w.contents.mul(scalar(0.5)), s2.w.contents.mul(scalar(0.5)), 10]); // should be made more exact
      return o.sum().sub(d);
    }
    else if (t1 === "Rectangle" && linelike([t2])) {
      const lpts = linePtsT(s2);
      const cp = closestptPtSeg(center(s1), lpts[0], lpts[1]);
      const d = dist(center(s1), cp);
      const o = s1.w.contents.div(scalar(2.0));    // TODO - make more exact (account for tall skinny boxes) or remove
      return o.sub(d); 
    }
    else if (t1 === "Curve" && t2 === "Rectangle") {
      const ret = zero;
      let d;
      let o;
      
      s1.pathData.contents.forEach((pt: number)=> {
        const tensorpt = tensor(pt);
        if (s2.w.contents.greater(s2.h.contents)) {
          d = dist(tensorpt, center(s2));
          o = s2.w.contents.mul(scalar(0.5)).add(scalar(10)); // TODO : FIX - this needs to take into account height - same with case above
          ret.add(o.sub(d));
        }
        else {
          d = dist(tensorpt, center(s2));
          o = s2.h.contents.mul(scalar(0.5)).add(scalar(10)); // TODO : FIX
          ret.add(o.sub(d));
        }
      });
      return ret;
    }
     else throw new Error(`${[t1, t2]} not supported for disjoint`);
  },

  smallerThan: ([t1, s1]: [string, any], [t2, s2]: [string, any]) => {
    // s1 is smaller than s2
    const offset = scalar(0.4).mul(s2.r.contents); // take 0.4 as param
    return s1.r.contents.sub(s2.r.contents).sub(offset);
  },

  outsideOf: (
    [t1, s1]: [string, any],
    [t2, s2]: [string, any],
    padding = 10.0
  ) => {
    if (t1 === "Text" && t2 === "Circle") {
      const textR = maximum(s1.w.contents, s1.h.contents);
      const d = dist(center(s1), center(s2));
      return s2.r.contents
        .add(textR)
        .add(padding)
        .sub(d);
    } else throw new Error(`${[t1, t2]} not supported for outsideOf`);
  },

  overlapping: (
    [t1, s1]: [string, any],
    [t2, s2]: [string, any],
    padding = 10
  ) => {
    if (t1 === "Circle" && t2 === "Circle") {
      return looseIntersect(center(s1), s1.r.contents,
        center(s2), s2.r.contents, padding);
    } else throw new Error(`${[t1, t2]} not supported for overlapping`);
  },

};

// -------- Helpers for writing objectives

const typesAre = (inputs: string[], expected: string[]): boolean =>
  (inputs.length === expected.length) && _.zip(inputs, expected).every(([i, e]) => i === e);

const linelike = (types: string[]): boolean => 
  (types.every((typ) => typ === "Arrow" || typ === "Line"));

// -------- (Hidden) helpers for objective/constraints/computations

const centerArrow2 = (arr: any, center1: Tensor, center2: Tensor, [o1, o2]: Tensor[]): Tensor => {
  const vec = center2.sub(center1); // direction the arrow should point to
  const dir = normalize(vec);

  let start = center1;
  let end = center2;

  // TODO: take in spacing, use the right text dimension/distance?, note on arrow directionality
  if (norm(vec).greater(o1.add(abs(o2)))) {
    start = center1.add(o1.mul(dir));
    end = center2.add(o2.mul(dir));
  }

  const fromPt = stack([arr.startX.contents, arr.startY.contents]);
  const toPt = stack([arr.endX.contents, arr.endY.contents]);

  return distsq(fromPt, start).add(distsq(toPt, end));
}

// returns midpoints of rectangle sides from left side clockwise
const sideMidpts = (rect: any): Array<Tensor> => {
  const ls = stack([rect.x.contents.sub(rect.w.contents.div(scalar(2.0))), rect.y.contents]); // left side
  const ts = stack([rect.x.contents, rect.y.contents.add(rect.h.contents.div(scalar(2.0)))]); // top side
  const rs = stack([rect.x.contents.add(rect.w.contents.div(scalar(2.0))), rect.y.contents]); // right side
  const bs = stack([rect.x.contents, rect.y.contents.sub(rect.h.contents.div(scalar(2.0)))]); // bottom side
  return [ls, ts, rs, bs];
}



// -------- Utils for objective/constraints/computations

const sc = (x: any): number => x.dataSync()[0];
const scs = (xs: any[]) => xs.map((e) => sc(e));

export const zero: Tensor = scalar(0);

// to prevent 1/0 (infinity). put it in the denominator
export const epsd: Tensor = scalar(10e-10);

export const looseIntersect = (center1: Tensor, r1: Tensor, center2: Tensor, r2: Tensor, padding: number) =>
  dist(center1, center2).sub(r1.add(r2).sub(scalar(padding)));
// dist (x1, y1) (x2, y2) - (s1 + s2 - 10)

export const center = (props: any): Tensor =>
  stack([props.x.contents, props.y.contents]); // HACK: need to annotate the types of x and y to be Tensor

// same as arrowPts in Evaluator, but returns arr of tensors rather than numbers
export const linePtsT = (props: any) : Tensor[] => [stack([props.startX.contents, props.startY.contents]), stack([props.endX.contents, props.endY.contents])];


export const dist = (p1: Tensor, p2: Tensor): Tensor => p1.sub(p2).norm();

// gives slope of line or arrow given start and end points
export const slope = (startX: Tensor, startY: Tensor, endX: Tensor, endY: Tensor): Tensor => {
  return (endY.sub(startY)).div(endX.sub(startX));
}

// finds closest point on line seg defined by start and end to any point pt
// https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
export const closestptPtSeg = (pt: Tensor, start: Tensor, end: Tensor): Tensor => {
  const lensq = distsq(start, end);
  const eps = Math.pow(10, -3);
  const dir = end.sub(start);
  if (lensq.less(scalar(eps)).dataSync()[0]) {
    // line seg is basically a point
    return start;
  }
  else {
    const t = (pt.sub(start)).dot(dir).div(lensq);
    const ct = maximum(zero, minimum(scalar(1.0), t));
    console.log("flag");
    print (start.add(ct.mul(dir)));
    return start.add(ct.mul(dir)); // walk along vector of line seg
  }
}

// takes a rect and a point and places the point on the perimeter of the rect
export const touchPerim = (s1: any, touchx : Tensor, touchy: Tensor) => {
  const xrange = inRange(touchx, s1.x.contents.sub(s1.w.contents.div(2.0)).sub(1.0), s1.x.contents.add(s1.w.contents.div(2.0)).add(1.0));
  const yrange = inRange(touchy, s1.y.contents.sub(s1.h.contents.div(2.0)).sub(1.0), s1.y.contents.add(s1.h.contents.div(2.0)).add(1.0));
  const d = dist(center(s1), stack([touchx, touchy]));
  const o = s1.w.contents.mul(scalar(0.5)); // should be made more exact
  // return xrange.add(yrange);
  return xrange.add(yrange).add(o.sub(d));
}

export const inRange = (a: Tensor, l: Tensor, r: Tensor) : Tensor => {
  if (a.less(l)) return a.sub(l).square()
  else if (a.greater(r)) return a.sub(r).square()
  else return zero;
}

// Be careful not to use element-wise operations. This should return a scalar.
// Apparently typescript can't check a return type of `Tensor<Rank.R0>`?
export const distsq = (p1: Tensor, p2: Tensor): Tensor => {
  const dp = p1.sub(p2);
  return dp.dot(dp);
}


// with epsilon to avoid NaNs
export const normalize = (v: Tensor): Tensor => v.div(v.norm().add(epsd));

// TODO: use it
// const getConstraint = (name: string) => {
//   if (!constrDict[name]) throw new Error(`Constraint "${name}" not found`);
//   // TODO: types for args
//   return (...args: any[]) => toPenalty(constrDict[name]);
// };
