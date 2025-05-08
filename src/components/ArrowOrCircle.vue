<template>
  <!-- 移动箭头（全局渲染） -->
  <div v-if="action.type === 'move' && global" style="position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; z-index: 10;">
    <svg
      :width="boardWidth"
      :height="boardHeight"
      style="position: absolute; left: 0; top: 0; width: 100%; height: 100%;"
    >
      <defs>
        <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L6,3 L0,6 L1.5,3 Z" fill="#f00"/>
        </marker>
      </defs>
      <line
        :x1="fromCenterX" :y1="fromCenterY"
        :x2="toCenterX" :y2="toCenterY"
        stroke="#f00" stroke-width="4"
        marker-end="url(#arrow)"
      />
    </svg>
  </div>
</template>

<script setup>
import { computed } from 'vue'
const props = defineProps({
  action: Object,
  cellWidth: Number,
  cellHeight: Number,
  fromPos: Array,
  toPos: Array,
  cellPos: Array,
  boardWidth: Number,
  boardHeight: Number,
  global: Boolean
})

// 箭头起止点（全局）
const fromCenterX = computed(() => (props.fromPos?.[1] + 0.5) * props.cellWidth)
const fromCenterY = computed(() => (props.fromPos?.[0] + 0.5) * props.cellHeight)
const toCenterX = computed(() => (props.toPos?.[1] + 0.5) * props.cellWidth)
const toCenterY = computed(() => (props.toPos?.[0] + 0.5) * props.cellHeight)

// // 动画控制
// const showAnim = ref(false)
// let animTimer = null
// watch(() => JSON.stringify(props.action), () => {
//   showAnim.value = false
//   if (animTimer) clearTimeout(animTimer)
//   // 下一个tick再加动画，保证class切换
//   setTimeout(() => {
//     showAnim.value = true
//     animTimer = setTimeout(() => {
//       showAnim.value = false
//     }, 2000)
//   }, 10)
// })
</script>
