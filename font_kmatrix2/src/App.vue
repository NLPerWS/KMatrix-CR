<template>
	<div id="app" >
        
        <div class="inputdiv">
            <h2 class="title" style="margin-left: 30px; color:cadetblue;">
                Please input external knowledge
            </h2>

            <textarea class="input-textarea" v-model="inputText">


            </textarea>
            
            <el-button  class="el-icon-s-promotion" type="primary" style="margin-top: 10px; margin-right: 20px; float: right;" @click="do_chat" :loading="submit_loading">
                Submit
            </el-button>


            <el-upload style="margin-left: 20px;"
            class="upload-demo"
            action="https://jsonplaceholder.typicode.com/posts/"
            :show-file-list="false"
            multiple
            :on-success="onSuccess"
            
            >
            <el-button  class="el-icon-upload" type="primary" style="margin-top: 10px; margin-right: 20px; float: right;" @click="do_chat">
                Upload Dataset And Test
            </el-button>
            </el-upload>

        </div>

        <div class="optiondiv">
            <h2 class="title" style="margin-left: 30px; color:cadetblue;">
                Option
            </h2>

            <!-- option -->
            <div style="width: 85%; margin-left: 20px; display: flex; flex-direction: column;" > 

                <div style="margin-top: 0px;">
                    <span class="demonstration">Model Name</span>
                    <div style="margin-top: 10px;"></div>
                    <el-select v-model="option_params.model_name" placeholder="Please select model_name">
                        <el-option
                        v-for="item in model_name_list"
                        :key="item"
                        :label="item"
                        :value="item">
                        </el-option>
                    </el-select>
                </div>

                <div style="margin-top: 10px;">
                    <span class="demonstration">Temperature</span>
                    <!-- <div style="margin-top: 10px;"></div> -->
                    <el-slider v-model="option_params.temperature" :step="0.1" :max="1" ></el-slider>
                </div>

                <div style="margin-top: 10px;">
                    <span class="demonstration">Max Tokens</span>
                    <!-- <div style="margin-top: 10px;"></div> -->
                    <el-slider v-model="option_params.max_tokens" :max="4096"></el-slider>
                </div>

                <div style="margin-top: 10px;">
                    <span class="demonstration">Conflict Avoidance Strategy</span>
                    <div style="margin-top: 10px;"></div>
                    <el-cascader placeholder="Please select IC Method"
                        v-model="option_params.root_conflict_avoidance_strategy"
                        :options="root_conflict_avoidance_strategy_list">
                    </el-cascader>
                </div>

                <!-- <div style="margin-top: 10px;">
                    <span class="demonstration">CM Conflict Avoidance Strategy</span>
                    <div style="margin-top: 5px;"></div>
                    <el-cascader placeholder="Please select CM Method"
                        v-model="option_params.cm_conflict_avoidance_strategy"
                        :options="cm_conflict_avoidance_strategy_list">
                    </el-cascader>
                </div>

                <div style="margin-top: 10px;">
                    <span class="demonstration">IC Conflict Avoidance Strategy</span>
                    <div style="margin-top: 5px;"></div>
                    <el-cascader placeholder="Please select IC Method"
                        v-model="option_params.ic_conflict_avoidance_strategy"
                        :options="ic_conflict_avoidance_strategy_list">
                    </el-cascader>
                </div>

                <div style="margin-top: 10px;">
                    <span class="demonstration">IM Conflict Avoidance Strategy</span>
                    <div style="margin-top: 5px;"></div>
                    <el-cascader placeholder="Please select IM Method"
                        v-model="option_params.im_conflict_avoidance_strategy"
                        :options="im_conflict_avoidance_strategy_list">
                    </el-cascader>
                </div> -->


            </div>


        </div>

        <div class="outputdiv">
            <h2 class="title" style="margin-left: 30px; color:cadetblue;">
                Result
            </h2>

            <div v-text="outputText" class="output-text">


            </div>


        </div>
	</div>
</template>

<script>
import { axios_instance } from "@/axios/index";

export default {
	data() {

        const model_name_list = [
            "Llama-2-7b-chat-hf",
        ];

        const cm_conflict_avoidance_strategy_list = [
            {
                value: "Faithful to Context",
                label: "Faithful to Context",
                children: [
                    {
                        value: "Context-Faithful",
                        label: "Context-Faithful"
                    },
                    {
                        value: "Aware-Decoding",
                        label: "Aware-Decoding"
                    },
                    {
                        value: "Contrastive-Decoding",
                        label: "Contrastive-Decoding"
                    },
                ]
            },
            {
                value: "Faithful to Memory",
                label: "Faithful to Memory",
                children: [
                    {
                        value: "llms-believe-the_earth_is_flat",
                        label: "llms-believe-the_earth_is_flat"
                    },
                ]
            },
            {
                value: "Disentangling Sources",
                label: "Disentangling Sources",
                children: [
                    {
                        value: "Disent QA",
                        label: "Disent QA"
                    },
                ]
            },
            {
                value: "Improving Factuality",
                label: "Improving Factuality",
                children: [
                    {
                        value: "Coiecd",
                        label: "Coiecd"
                    },
                ]
            },



        ];
        const ic_conflict_avoidance_strategy_list = [
            {
                value: "Improving Factuality",
                label: "Improving Factuality",
                children: [
                    {
                        value: "ICL-whole",
                        label: "ICL-whole"
                    },
                    {
                        value: "ICL-seprate",
                        label: "ICL-seprate"
                    },
                    {
                        value: "Factools",
                        label: "Factools"
                    },

                ]
            },
        ];
        const im_conflict_avoidance_strategy_list = [
            {
                value: "Improving Factuality",
                label: "Improving Factuality",
                children: [
                    {
                        value: "Dola",
                        label: "Dola"
                    },
                    {
                        value: "Concord",
                        label: "Concord"
                    },
                ]
            },
        ];

        const root_conflict_avoidance_strategy_list = [
            {
                value: "None",
                label: "None",
                leaf: true
            },
            {
                value: "CM Conflict Resolution",
                label: "CM Conflict Resolution",
                children: cm_conflict_avoidance_strategy_list
            },
            {
                value: "IC Conflict Resolution",
                label: "IC Conflict Resolution",
                children: ic_conflict_avoidance_strategy_list
            },
            {
                value: "IM Conflict Resolution",
                label: "IM Conflict Resolution",
                children: im_conflict_avoidance_strategy_list
            },

        ]


        return {
            submit_loading:false,
            
            model_name_list: model_name_list,
            root_conflict_avoidance_strategy_list:root_conflict_avoidance_strategy_list,
            // cm_conflict_avoidance_strategy_list : cm_conflict_avoidance_strategy_list,
            // ic_conflict_avoidance_strategy_list : ic_conflict_avoidance_strategy_list,
            // im_conflict_avoidance_strategy_list : im_conflict_avoidance_strategy_list,

            inputText: '',
            option_params: {
                "model_name": model_name_list[0],
                "temperature": 0.1,
                "max_tokens": 1000,
                "root_conflict_avoidance_strategy":root_conflict_avoidance_strategy_list[0]['value']
                // "cm_conflict_avoidance_strategy":cm_conflict_avoidance_strategy_list[0]['value'],
                // "ic_conflict_avoidance_strategy":ic_conflict_avoidance_strategy_list[0]['value'],
                // "im_conflict_avoidance_strategy":im_conflict_avoidance_strategy_list[0]['value'],
            },
            outputText:'',


		};
	},


	methods: {

        onSuccess(response, file, fileList){
            // console.log("🚀 -> fileList:\n", fileList)
            // console.log("🚀 -> file:\n", file)
            // console.log("🚀 -> response:\n", response)
            let str = ""
            for(let i in fileList){
                str += fileList[i].name + "\n"
            }
            this.inputText = str;
            this.$message({
                message: 'Upload success!',
                type: 'success',
                duration:2500
            });
        },

        do_chat() {

            console.log("🚀 -> this.option_params:\n", this.option_params)
            // console.log("🚀 -> this.inputText:\n", this.inputText)
            // console.log("🚀 -> this.outputText:\n", this.outputText)

            if (this.inputText == "") {
                this.$message({
                    message: 'Please enter the question',
                    type: 'warning',
                    duration:2500
                });
                return;
            }

            let chat_awit = this.$message({
                message: 'Running, please wait...',
                duration:0
            });
            this.submit_loading = true;
            this.outputText = '';

			let param = {
				input: this.inputText,
				option: this.option_params,
			};
			axios_instance
				.post("/chat", param)
				.then((res) => {
                    this.outputText = res.data.data;
                    chat_awit.close();
                    this.$message({
                        message: 'Run success!',
                        type: 'success',
                        duration:2500
                    });
                    this.submit_loading = false;
				})
				.catch((err) => {
                    this.outputText = err;
                    chat_awit.close();
                    this.$message({
                        message: 'error',
                        type: 'error',
                        duration:2500
                    });
                    this.submit_loading = false;
                });
        }


	},
};
</script>

<style>

/* 整个滚动条 */
::-webkit-scrollbar {
	width: 6px; /* 滚动条的宽度 */
	height: 6px; /* 滚动条的高度 */
}

/* 滚动条的轨道 */
::-webkit-scrollbar-track {
	background: #dcdfe6; /* 轨道的背景色 */
	border-radius: 10px; /* 轨道的圆角 */
}

/* 滚动条上的滑块 */
::-webkit-scrollbar-thumb {
	background: #CACAD0; /* 滑块的背景色 */
	border-radius: 10px; /* 滑块的圆角 */
}

/* 滑块在悬停时的样式 */
::-webkit-scrollbar-thumb:hover {
	background: #CACAD0; /* 悬停时滑块的背景色 */
}


</style>

<style scoped>

#app{
    display:flex; 
    flex-direction: row;
    flex-wrap: wrap;
}

.inputdiv {
    margin-left: 10px;
    /* 比 optiondiv 宽度多30px */
    width: calc(100% - 380px); 
    height: 400px;
    margin-top: 10px;
	border-radius: 10px; /* 圆角 */
	box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1); /* 阴影效果 */
	transition: transform 0.3s, box-shadow 0.3s; /* 过渡效果 */
	border-top: 0.4px solid #ebebeb; /* 上边框颜色和厚度 */
}
.input-textarea {
    /* 禁止拉伸 */
    resize: none;
    padding: 10px;
    overflow-y: auto;
    font-size: 22px;

    width: calc(100% - 50px);
    height: calc(100% - 180px);
    margin-top: 10px;
    margin-left: 10px;


    border: 1px solid #D9D9D9;
	border-radius: 10px; /* 圆角 */
	box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1); /* 阴影效果 */
	transition: transform 0.3s, box-shadow 0.3s; /* 过渡效果 */
	border-top: 0.4px solid #ebebeb; /* 上边框颜色和厚度 */

}

.optiondiv {
    margin-left: 15px;
    width: 350px;
    height: 400px;
    margin-top: 10px;
	border-radius: 10px; /* 圆角 */
	box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1); /* 阴影效果 */
	transition: transform 0.3s, box-shadow 0.3s; /* 过渡效果 */
	border-top: 0.4px solid #ebebeb; /* 上边框颜色和厚度 */
}

.demonstration{
    color: #8492BA;
}




.outputdiv{
    margin-left: 10px;
    margin-top: 10px;
    width: calc(100% - 15px);
    height: calc(100vh - 440px);
    margin-top: 10px;
	border-radius: 10px; /* 圆角 */
	box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1); /* 阴影效果 */
	transition: transform 0.3s, box-shadow 0.3s; /* 过渡效果 */
	border-top: 0.4px solid #ebebeb; /* 上边框颜色和厚度 */

}

.output-text{

    width: calc(100% - 50px);
    padding: 10px;
    margin-left: 15px;
    height: calc(100% - 110px);

    font-size: 22px;
    color: darkcyan;
    overflow-y: auto;

    word-wrap: break-word; /* 旧的属性名 */
    overflow-wrap: break-word; /* 新的属性名，推荐使用 */    
    white-space: pre-wrap; /* 保留换行符和空白符 */

	border-radius: 10px; /* 圆角 */
	box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1); /* 阴影效果 */
	transition: transform 0.3s, box-shadow 0.3s; /* 过渡效果 */
	border-top: 0.4px solid #ebebeb; /* 上边框颜色和厚度 */

}

</style>
