  handshake.func @if_loop_1 (%a : memref<?xi32>, %0 : i32, %65 : none) -> (i32, none) {
    %1 = constant %83 {value = 0 : i32} : i32       // cst_0
    %2 = arith.cmpi sgt, %25, %1 : i32        // icmp_0
    %3 = constant %74 {value = 1 : i1} : i1       // brCst_block2
    %4 = constant %71 {value = 0 : i32} : i32       // cst_1
    %5 = mux %94 [%42, %52] : index, i32        // phi_3
    %6 = constant %72 {value = 0 : i32} : i32       // cst_2
    %7 = mux %95 [%44, %54] : index, i32        // phi_4
    %load_source = source
    %8, %9 = load [%28] %62, %load_source : i32, i32        // load_7
    %10 = constant %84 {value = 5 : i32} : i32  // cst_3
    %11 = arith.muli %8, %10 : i32      // mul_8
    %12 = constant %85 {value = 10 : i32} : i32 // cst_4
    %13 = arith.cmpi sgt, %29, %12 : i32        // icmp_9
    %14 = constant %86 {value = 0 : i32} : i32    // cst_5
    %15 = mux %13 [%14, %30] : i1, i32        // select_0
    %16 = arith.addi %88, %15 : i32   // add_10
    %17 = constant %87 {value = 1 : i32} : i32      // cst_6
    %18 = mux %82 [%36, %53] : index, i32       // phi_14
    %19 = arith.addi %27, %17 : i32   // add_11
    %20 = arith.cmpi slt, %31, %33 : i32      // icmp_12
    %21 = constant %66 {value = 0 : i32} : i32      // cst_7
    %23 = merge %37 : i32     // phi_n0
    %24 = merge %46, %56 : i32        // phi_n1
    %25, %26 = fork [2] %0 : i32      // fork_0
    %27, %28 = fork [2] %89 : i32     // fork_1
    %29, %30 = fork [2] %11 : i32       // fork_2
    %31, %32 = fork [2] %19 : i32     // fork_4
    %33, %34 = fork [2] %91 : i32     // fork_5
    %35, %36 = cond_br %40, %21 : i32 // branch_0
    %37, %38 = cond_br %39, %26 : i32 // branch_1
    %39, %40, %41 = fork [3] %2 : i1 // fork_6
    %42, %43 = cond_br %50, %4 : i32  // branch_2
    %44, %45 = cond_br %49, %6 : i32  // branch_3
    %46, %47 = cond_br %48, %23 : i32 // branch_4
    %48, %49, %50, %51 = fork [4] %3 : i1    // fork_7
    %52, %53 = cond_br %60, %16 : i32 // branch_5
    %54, %55 = cond_br %59, %32 : i32 // branch_6
    %56, %57 = cond_br %58, %34 : i32 // branch_7
    %58, %59, %60, %61 = fork [4] %20 : i1   // fork_8
    %62, %63 = extmemory [ld=1, st=0] (%a : memref<?xi32>) (%9) {id = 0 : i32} : (i32) -> (i32, none)     // MC_a
    %66, %67 = fork [2] %65 : none     // forkC_10
    %68, %69 = cond_br %41, %67 : none // branchC_8
    %70 = merge %68 : none     // phiC_2
    %71, %72, %73, %74 = fork [4] %70 : none   // forkC_11
    %75, %76 = cond_br %51, %73 : none // branchC_9
    %77, %78 = control_merge %75, %79 : none    // phiC_3
    %79, %80 = cond_br %61, %92 : none // branchC_10
    %81, %82 = control_merge %69, %80 : none    // phiC_4
    sink %93 : none     // sink_0
    sink %35 : i32    // sink_1
    sink %38 : i32      // sink_2
    sink %43 : i32       // sink_3
    sink %45 : i32       // sink_4
    sink %47 : i32      // sink_5
    sink %55 : i32      // sink_6
    sink %57 : i32      // sink_7
    sink %76 : none     // sink_8
    %83 = source        // source_0
    %84 = source        // source_1
    %85 = source        // source_2
    %86 = source        // source_3
    %87 = source        // source_4
    %88 = buffer [2] %5 {sequential = true} : i32     // buffI_0
    %89 = buffer [2] %7 {sequential = true} : i32     // buffI_1
    %90 = buffer [2] %18 {sequential = true} : i32      // buffI_2
    %91 = buffer [2] %24 {sequential = true} : i32    // buffA_3
    %92 = buffer [2] %77 {sequential = true} : none    // buffA_4
    %93 = buffer [2] %81 {sequential = true} : none     // buffA_5
    %94, %95 = fork [2] %78 : index     // fork_14
    return %90, %63 : i32, none    // ret_0
  }