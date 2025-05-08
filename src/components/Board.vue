<template>
  <div class="relative" :style="{ width: boardWidth + 'px', height: boardHeight + 'px' }">
    <div
      v-for="(row, i) in 14"
      :key="i"
      class="flex flex-row"
      :style="{ height: cellHeight + 'px' }"
    >
      <div
        v-for="(col, j) in 10"
        :key="j"
        style="display: flex; flex-direction: row; position: relative;"
        :style="{ width: cellWidth + 'px', height: cellHeight + 'px' }"
      >
        <img
          :src="getCellImg(i, j)"
          :style="{ width: cellWidth + 'px', height: cellHeight + 'px' }"
          class=""
        />
        <!-- 只渲染红框 -->
        <div
          v-if="shouldShowMark(i, j)"
          class="absolute click-border"
          :style="{
            left: 0,
            top: 0,
            width: cellWidth + 'px',
            height: cellHeight + 'px',
            zIndex: 20
          }"
        ></div>
      </div>
    </div>
    <!-- 全局箭头渲染（move类型） -->
    <ArrowOrCircle
      v-if="currentAction && currentAction.type === 'move'"
      :action="currentAction"
      :cellWidth="cellWidth"
      :cellHeight="cellHeight"
      :fromPos="currentAction.from"
      :toPos="currentAction.to"
      :boardWidth="boardWidth"
      :boardHeight="boardHeight"
      global
    />
  </div>
</template>

<script setup>
import ArrowOrCircle from './ArrowOrCircle.vue'
import { computed } from 'vue'

const props = defineProps({
  board: Array,
  actions: [Array, Object],
  currentStep: Number,
  typeImgs: Object,
  bgImg: String
})

const cellWidth = 50
const cellHeight = 52
const boardWidth = cellWidth * 10
const boardHeight = cellHeight * 14

function getCellImg(i, j) {
  const type = props.board?.[i]?.[j] || 0
  if (type === 0) return props.bgImg
  return props.typeImgs[type] || props.bgImg
}

const currentAction = computed(() => {
  if (Array.isArray(props.actions)) {
    return props.actions[props.currentStep] || {}
  }
  return props.actions || {}
})

function shouldShowMark(i, j) {
  const action = currentAction.value
  if (!action || !action.type) return false
  if (action.from && action.from[0] === i && action.from[1] === j) {
    return true
  }
  if (action.match_cell && action.match_cell[0] === i && action.match_cell[1] === j) { 
    return true
  }
  return false
}
</script>

<style scoped>
.click-border {
  box-sizing: border-box;
  border: 3px solid #f00;
  border-radius: 8px;
  width: 100%;
  height: 100%;
  pointer-events: none;
}
</style>
