/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief lower_atomic.cc
 */

#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/transform.h>

#include <set>
#include <utility>

#include "../../support/utils.h"
#include "../schedule/analysis.h"
#include "ir_utils.h"

namespace tvm {

namespace tir {

class EliminateStoreBuffer : public ExprMutator {
 public:
  explicit EliminateStoreBuffer(Buffer store_buffer, Array<PrimExpr> store_indices) : store_buffer_(store_buffer), store_indices_(store_indices) {}
  std::unordered_set<const BufferLoadNode*> load_buffer_list;

 private:

  PrimExpr VisitExpr_(const AddNode* op) final {
    if (op->a.as<BufferLoadNode>()) {
      auto buffer_load = Downcast<BufferLoad>(op->a);
      if(buffer_load->buffer.same_as(store_buffer_)) {
        return Add(Cast(buffer_load->dtype, Integer(0)),VisitExpr(op->b),op->span);
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

  Buffer store_buffer_;
  Array<PrimExpr> store_indices_;
};

class LowerAtomicTransformer : public StmtExprMutator {
 public:
  explicit LowerAtomicTransformer() : is_atomic_block(false) {}

 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    if (op->annotations.Get("atomic").defined()) {
      is_atomic_block = true;
    }
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    is_atomic_block = false;
    return ret;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (is_atomic_block) {
      auto helper = EliminateStoreBuffer(op->buffer, op->indices);
      auto value = helper(op->value);
      return Evaluate(atomic_add(op->buffer->data, op->indices[0],
                                 value));  // cuda atomic add op->buffer[op->indices]+=op->value
    } else {
      return GetRef<BufferStore>(op);
    }
  }

  bool is_atomic_block;
};

PrimFunc LowerAtomic(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = LowerAtomicTransformer()(std::move(fptr->body));
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass LowerAtomic() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerAtomic(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerAtomic", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerAtomic").set_body_typed(LowerAtomic);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
