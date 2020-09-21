## TODO

- Generalize tests
  - Test at a couple of different points
  - Factor out graph building code from test1
- Convert test2 to check grad results programmatically
- Check the tests
- Look at more concise generated code

               INPUTS           OUTPUTS
    Energy   | known          | only 1
    Gradient | ENERGY inputs  | grad nodes (attached to the original energy input nodes)

>>> Get gradGraph4 to work
    Put gradGraph 4 back in the test
    Evaluate ifcond and gt
    Where did the Math.max go?
    Is there a problem with evalEnergyOnCustom?
    for grad of max, needs to support IfCond generation / evaluation

>>> Also revert the sameCenter thing, test fully
>>> Also revert the objfn stuff, test fully

- Check grad results for programmatic function
  remove throw Error in `energyAndGradDynamic`
      also rename this function

- Write more hardcoded tests

- Lay out diagram
  - Only compile gradient and energy once, in the beginning, and store it
  - Profile this
  - Use it in the opt and line search

======

- account for the weights of the optimization as parameters to the function, and partially apply it
- add back the constraints in venn-small.sty
  - fix n-ary problem
- lay out and benchmark

- figure out why line search is behaving oddly on venn-small.sty

- make this work for resampling

===

revert sameCenter and venn-small.sty again
swap default hasWeight from false to true

TODO: fix the design for hyperparameters

- types:
  GradGraphs should have a field for hyperparameters, 
  weight: Maybe<VarAD>

- to generate fns: IF there is a weight,
  hyperparameters are just treated as inputs that come before all the other ones
      f0(x0, x1...xn) <-- x0 might be a hyperparameter, 
        { x_{n+1} = ... x0 + x1 ... } <-- the body of the function might refer to the hyperparameter just like a normal input
      similar for grads:
      gradf0(x0, x1...) <-- x0 might be a hyperparameter, etc.

  then, each function is curried with its first argument only
      f(x0)(x1,...) = f0(x0, x1,...)
      gradf(x0)(x1,...) = f0(x0, x1,...)

  if no weight, just gen the fn w/ no currying

- to use: IF there is a weight,
  hyperparameters are treated as partial applications for both the energy and the gradient
  f(hs0)(xs0): number
  gradf(hs0)(xs0): number[]

  can (and should) be applied with different hyperparameters across EP runs

  f(hs1)(xs1)
  gradf(hs1)(xs1)

  if no weight, just apply the fn w/ no partial application

---

- make sure genCode deals with weight type
  get rid of hasWeight
  should it actually take graphs...
  and make a 'genGradient' wrapper?
- optimizer: store the partially applied fn in state
  - reapply with new weight in EP

---

works with `sameCenter`
works with `contains`, just doesn't satisfy `maxSize`

TODO: some combination of them results in a NaN

does work instantly to lay it out

TODO: Don't resynthesize code on resample

TODO: Check for convergence in inner loop

TODO: narrow down the NaN

we are in stepBasic not stepEP; rename functions to make it clearer

TODO: Re-profile this
TODO: Put in example synthesized code
TODO: Fix failing constraint tests -- due to number magnitude
TODO: Post results

TODO: Port tree.sty

---

works with `runpenrose set-theory-domain/tree.sub set-theory-domain/venn-opt-test.sty set-theory-domain/setTheory.dsl`

TODO: Fix that the resample hack breaks on switching examples since it saves the cached functions...
TODO: Also breaks if you resample without generating the function on first sample. Clearly this should be part of the state
TODO: The code below (the line) hasn't been ported to web-perf yet

TODO: (for tree.sty)
- repel (with padding) X
- centerArrow
- above
- equal

---

TODO: Document new available functions

Tree repel
    Need inverse

TODO: Document what I need to do to add a new operation
// ADDING A NEW OP: (TODO: document further)
// Add its definition above
// Add it to the opMap
// Add its js mapping (code) to traverseGraph

---

Tree works instantly. TODO: Document the GIF 
TODO: Also finds another symmetric result which TF didn't find earlier: https://www.dropbox.com/s/ef6gzkmra1zsa9q/Screen%20Shot%202020-09-05%20at%2011.38.49%20PM.png?dl=0
(does this satisfy the style program?)

---

TODO 9/14/20

- Port euclidean.sty
- Clean up web-perf code
- Look at what it takes to merge with web-runtime

## Port euclidean.sty

- Have to do it incrementally

Constraints
- atDist
- contains (rect, circle/rect, padding)
- perpendicular (3 tups)

Objectives
- repel (line, rect, weight)
- repel (circle, circle, weight)
- repel (line, rect, weight)
- repel (path, rect, weight)

---

- Need to evaluate lists, tuples

- Need to write evalExprs as a fold
  - nvm, evalExpr doesn't update the translation, and this is now run only once anyway

- Is the NaN problem that variable names don't work for fields?
  `variables:  (2) ["a.x.", "a.y."]`
No, actually the variable names don't matter because we only use JSON.stringify

Why didn't it catch the error in `get`'s type?

TODO: How to set up tide electric indent (C-j)? Enter seems to work, when all lines at end of ~/.emacs are commented out

- Write min X
- Write lt  X
- Write contains on rect-circ X
- Write atDist on labels
  Have to get polygon of text

- Write repel on labels

TODO 9/15/20

- can we even render beziers now?
- This style all requires inline computations
- Ok, I should merge w/ web-runtime
- will require implementing tensor-to-number conversion for paths 

Merge:
- Need to combine web-perf's use of DiffVar (but really, just VarAD) with web-runtime's factoring-out of numbers/autodiff from evaluations
- Probably need to change use of `scalar`, etc. (number <-> Tensor conversion) to number <-> varAD conversion
- Need to fix the linear-algebra-js dependency as suggested on slack

- Is numOf even necessary? I thought that the compiled stuff is just going to set the varying state as NUMBERS not variables
- Evaluator needs to use `VarAD` everywhere, not tensors

    // TODO: Evaluating the shapes for display is still done via interpretation on VarADs; not compiled

Check type of input in walkTranslationCovnert in EngineUtils

Distinguish all uses of varOf with constOf

TODO: throw Error("Pow op unimplemented");

Need to rename stepBasic, minimizeBasic, converged2, etc. to the right names

Delete or convert all the *old functions in constraints

Now:
- Tree works
- Geometry works

  TODO: there is a linalg error, maybe when the frontend is running but the running example (backend) changes
      > Unhandled Rejection (BindingError): Cannot pass deleted object as a pointer of type DenseMatrix*

- Venn steps but line search gets stuck
  step by step, you can see the line search ends w/ "stopping early"
  grad norm is still small 
  autostep: the browser is locked up, so much that the promise fails? why?
      > tslib.es6.js:94 Uncaught (in promise) 2130705432 - Exception catching is disabled, this exception cannot be caught. Compile with -s DISABLE_EXCEPTION_CATCHING=0 or DISABLE_EXCEPTION_CATCHING=2 to catch.

Did some objective/constraint become contradictory?

Overall, are there other obvious things that I missed in the merge?

venn-small doesn't work

could it be a state problem w/ the frontend?
works ok with the small sets example

tree-debug.sub + modified venn-opt-test.sty

Not sure why `contains` is acting up? Or just the combination of them?

Could it be uoStop? Yes (but doesn't rule out line search acting up in the future, for harder problems)

---

Still need to clean up Autodiff, Constraints, Evaluator, Optimizer

---

TODO: Does the new autodiff work on function calls and inline computations in Style??
TODO: Clean up Computations.ts
TODO: Port an example that "pushes back" thru a computation?