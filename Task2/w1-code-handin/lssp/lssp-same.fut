-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input {
--    [1, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1]
-- }
-- output {
--    5
-- }

-- ==
-- compiled input {
--    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, -6, 1]
-- }
-- output {
--    7
-- }

-- ==
-- compiled input {
--    [1, -2, -2, -2, -2, -2, -2, 0, 3, 4, -6, 1]
-- }
-- output {
--    6
-- }

import "lssp"
import "lssp-seq"

type int = i32

let main (xs: []int) : int =
  let pred1 _   = true
  let pred2 x y = (x == y)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs
