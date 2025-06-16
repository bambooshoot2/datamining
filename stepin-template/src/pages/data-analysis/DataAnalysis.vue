<template>
  <div class="p-4 max-w-xl mx-auto">
    <h2 class="text-xl font-bold mb-6 text-center">糖尿病数据分析</h2>
    <form @submit.prevent="predictDiabetes" class="space-y-6">

      <div class="grid grid-cols-2 gap-6">

        <!-- 第一行：性别 -->
        <div>
          <label class="block mb-1 text-sm font-medium text-gray-700">性别</label>
          <select v-model="form.gender" required
                  class="w-full border border-gray-300 rounded px-2 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
            <option value="1">男</option>
            <option value="0">女</option>
            <option value="2">其他</option>
          </select>
        </div>

        <!-- 第一行：年龄 -->
        <div>
          <label class="block mb-1 text-sm font-medium text-gray-700">年龄</label>
          <input v-model.number="form.age" type="number" required
                 class="w-full border border-gray-300 rounded px-2 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500" />
        </div>

        <!-- 第二行：高血压 (占两列宽) -->
        <div class="col-span-2">
          <label class="block mb-1 text-sm font-medium text-gray-700">是否患有高血压</label>
          <select v-model.number="form.hypertension" required
                  class="w-full border border-gray-300 rounded px-2 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
            <option :value="0">否</option>
            <option :value="1">是</option>
          </select>
        </div>

        <!-- 第三行：心脏病 -->
        <div>
          <label class="block mb-1 text-sm font-medium text-gray-700">是否患有心脏病</label>
          <select v-model.number="form.heart_disease" required
                  class="w-full border border-gray-300 rounded px-2 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
            <option :value="0">否</option>
            <option :value="1">是</option>
          </select>
        </div>

        <!-- 第三行：吸烟史 -->
        <div>
          <label class="block mb-1 text-sm font-medium text-gray-700">吸烟史</label>
          <select v-model="form.smoking_history" required
                  class="w-full border border-gray-300 rounded px-2 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
            <option :value="4">never</option>
            <option :value="0">no info</option>
            <option :value="3">former</option>
            <option :value="5">not current</option>
            <option :value="2">ever</option>
            <option :value="1">current</option>
          </select>
        </div>

        <!-- 第四行：BMI (占两列宽) -->
        <div class="col-span-2">
          <label class="block mb-1 text-sm font-medium text-gray-700">BMI</label>
          <input v-model.number="form.bmi" type="number" step="0.1" required
                 class="w-full border border-gray-300 rounded px-2 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500" />
        </div>

        <!-- 第五行：HbA1c 和 血糖值 两个一行 -->
        <div>
          <label class="block mb-1 text-sm font-medium text-gray-700">HbA1c 值</label>
          <input v-model.number="form.HbA1c_level" type="number" step="0.1" required
                 class="w-full border border-gray-300 rounded px-2 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500" />
        </div>

        <div>
          <label class="block mb-1 text-sm font-medium text-gray-700">血糖值</label>
          <input v-model.number="form.blood_glucose_level" type="number" required
                 class="w-full border border-gray-300 rounded px-2 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500" />
        </div>

      </div>
      <!-- 提交按钮 -->
      <div>
        <button
            type="submit"
            class="w-full py-2 bg-blue-600 hover:bg-blue-700 text-white rounded">
            数据分析
        </button>
      </div>
      <div v-if="healthMessage || warnings.length" class="mt-4 p-4 bg-yellow-100 rounded border border-yellow-400">
      <p class="font-bold text-yellow-800">{{ healthMessage }}</p>
      <ul class="list-disc ml-6 text-yellow-700">
        <li v-for="(w, i) in warnings" :key="i">{{ w }}</li>
      </ul>
    </div>
    </form>
  </div>
</template>

<script>
export default {
  data() {
    return {
      form: {
        gender: '',
        age: null,
        hypertension: 0,
        heart_disease: 0,
        smoking_history: 0,
        bmi: null,
        HbA1c_level: null,
        blood_glucose_level: null
      },
      warnings: [],
      healthMessage: ''
    };
  },
  methods: {
    predictDiabetes() {
      this.warnings = [];
      this.healthMessage = '';

      // 简单健康检查逻辑（你可以根据业务调整）
      const { age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level } = this.form;

      if (
        age > 0 &&
        hypertension === 0 &&
        heart_disease === 0 &&
        bmi >= 18.5 && bmi <= 24.9 &&
        HbA1c_level < 5.7 &&
        blood_glucose_level >= 70 && blood_glucose_level <= 140
      ) {
        this.healthMessage = "健康状态良好，请继续保持！";
      } else {
        // 可以加入详细 warning
        if (bmi < 18.5 || bmi > 24.9) this.warnings.push("BMI 超出正常范围");
        if (HbA1c_level >= 5.7) this.warnings.push("HbA1c 值偏高，注意血糖管理");
        else if(HbA1c_level>=200)this.warnings.push("HbA1c过高,很有可能患有糖尿病,请及时前往检查");
        if ((blood_glucose_level > 140&&blood_glucose_level<200))this.warnings.push("血糖值偏高");
        else if(blood_glucose_level>=200)this.warnings.push("血糖极高,很有可能患有糖尿病,请及时前往检查");
        if (hypertension === 1) this.warnings.push("患有高血压");
        if (heart_disease === 1) this.warnings.push("患有心脏病");
        this.healthMessage = "请注意以下健康提示：";
      }
    }
  }
}
</script>
