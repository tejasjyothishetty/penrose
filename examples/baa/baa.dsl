type Box
type Arr

predicate To : Box b1 * Box b2 * Arr a1

notation "A -C> B" ~ "To(A, B, C)" 