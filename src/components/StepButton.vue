<template>
  <button class="shadcn-btn" :disabled="disabled" @click="$emit('next-step')">
    下一步
  </button>
</template>

<script setup>
defineProps({
  disabled: Boolean
})
</script>
