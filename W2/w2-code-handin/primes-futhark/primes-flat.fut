-- Primes: Flat-Parallel Version
-- ==
-- compiled input { 30 } output { [2,3,5,7,11,13,17,19,23,29] }
-- compiled input { 10000000i32 } auto output

-- segmented scan with (+) on floats:
let sgmSum [n] (flg : [n]i32) (arr : [n]i32) : [n]i32 =
  let flgs_vals = 
    scan ( \ (f1, x1) (f2,x2) -> 
            let f = f1 | f2 in
            if f2 > 0 then (f, x2)
            else (f, x1 + x2) )
         (0,0) (zip flg arr)
  let (_, vals) = unzip flgs_vals
  in vals

let primesFlat (n : i32) : []i32 =
  let sq_primes   = [2,3,5,7]
  let len  = 8
  let (sq_primes, _) =
    loop (sq_primes, len) while len < n do
      -- this is "len = min n (len*len)" 
      -- but without running out of i32 bounds 
      let len = if n / len < len then n else len*len

      let mult_lens = map (\ p -> (len / p) - 1 ) sq_primes
      let flat_size = reduce (+) 0 mult_lens

      --------------------------------------------------------------
      -- The current iteration knowns the primes <= 'len', 
      --  based on which it will compute the primes <= 'len*len'
      -- ToDo: replace the dummy code below with the flat-parallel
      --       code that is equivalent with the nested-parallel one:

        --let composite = map (\ p -> let mm1 = (len / p) - 1
           --                          in  map (\ j -> j * p ) (map (+2) (iota mm1))
             --                ) sq_primes
         --let not_primes = reduce (++) [] composite
      --
      -- Your code should compute the right `not_primes`
      -- Please look at the lecture slides L2-Flattening.pdf to find
      --  the normalized nested-parallel version.
      -- Note that the scalar computation has already been distributed
      --  and the result is in "mult_lens", but it stores `len / p - 1`,
      --  where `p \in sq_primes`.
      --

      let mm1s = map (\p -> (len/p) - 1) sq_primes
      let shp_mm = scan (+) 0 mm1s 
      let shp = map(\i -> if i == 0 then 0 else shp_mm[i - 1]) (indices shp_mm)
      let zeros = replicate flat_size 0  
      let flags = scatter zeros shp sq_primes 
      let ones = replicate flat_size 1
      let seg = sgmSum flags ones 
      let mmp2 = map (+1) seg
      
      let flagones = map(\i -> if i == 0 then 0 else 1) flags 
      let scanflag = scan (+) 0 flagones
      let primes = map(\i -> sq_primes[i]) scanflag
      let not_primes = map2 (*) mmp2 primes 


      -- If not_primes is correctly computed, then the remaining
      -- code is correct and will do the job of computing the prime
      -- numbers up to n!
      --------------------------------------------------------------
      --------------------------------------------------------------

       let zero_array = replicate flat_size 0
       let mostly_ones= map (\ x -> if x > 1 then 1 else 0) (iota (len+1))
       let prime_flags= scatter mostly_ones not_primes zero_array
       let sq_primes = filter (\i-> i>1 && i<=n && prime_flags[i]>0i32) (0...len)

       in  (sq_primes, len)

  in sq_primes

-- RUN a big test with:
-- $ futhark opencl primes-flat.fut
-- $ echo "10000000" | ./primes-flat -t /dev/stderr -r 10 > /dev/null
let main (n : i32) : []i32 = primesFlat n
