/*
 * MIT License
 *
 * Copyright (c) 2023 Carnegie Mellon University
 *
 * This file is part of RoWild.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
*/

// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps

module main_scaled_fixed2ieee_63_1_Pipeline_2 (
    ap_clk,
    ap_rst,
    ap_start,
    ap_done,
    ap_idle,
    ap_ready,
    out_bits_2_1_reload,
    out_bits_1_1_reload,
    out_bits_0_1_reload,
    in_val,
    out_bits_2_2_out,
    out_bits_2_2_out_ap_vld,
    out_bits_1_2_out,
    out_bits_1_2_out_ap_vld,
    out_bits_0_21_out,
    out_bits_0_21_out_ap_vld
);

    parameter ap_ST_fsm_pp0_stage0 = 1'd1;

    input ap_clk;
    input ap_rst;
    input ap_start;
    output ap_done;
    output ap_idle;
    output ap_ready;
    input [31:0] out_bits_2_1_reload;
    input [31:0] out_bits_1_1_reload;
    input [31:0] out_bits_0_1_reload;
    input [62:0] in_val;
    output [31:0] out_bits_2_2_out;
    output out_bits_2_2_out_ap_vld;
    output [31:0] out_bits_1_2_out;
    output out_bits_1_2_out_ap_vld;
    output [31:0] out_bits_0_21_out;
    output out_bits_0_21_out_ap_vld;

    reg ap_idle;
    reg out_bits_2_2_out_ap_vld;
    reg out_bits_1_2_out_ap_vld;
    reg out_bits_0_21_out_ap_vld;

    (* fsm_encoding = "none" *) reg   [0:0] ap_CS_fsm;
    wire    ap_CS_fsm_pp0_stage0;
    wire    ap_enable_reg_pp0_iter0;
    reg    ap_enable_reg_pp0_iter1;
    reg    ap_idle_pp0;
    wire    ap_block_pp0_stage0_subdone;
    wire   [0:0] icmp_ln401_fu_142_p2;
    reg    ap_condition_exit_pp0_iter0_stage0;
    wire    ap_loop_exit_ready;
    reg    ap_ready_int;
    wire    ap_block_pp0_stage0_11001;
    reg   [1:0] i_2_reg_339;
    wire   [5:0] sub_ln404_fu_162_p2;
    reg   [5:0] sub_ln404_reg_346;
    wire   [0:0] icmp_ln403_fu_174_p2;
    reg   [0:0] icmp_ln403_reg_351;
    wire   [5:0] sub_ln403_2_fu_186_p2;
    reg   [5:0] sub_ln403_2_reg_357;
    wire   [5:0] sub_ln403_4_fu_206_p2;
    reg   [5:0] sub_ln403_4_reg_362;
    reg   [1:0] i_fu_58;
    wire   [1:0] i_3_fu_148_p2;
    wire    ap_loop_init;
    reg   [1:0] ap_sig_allocacmp_i_2;
    wire    ap_block_pp0_stage0;
    reg   [31:0] out_bits_0_21_fu_62;
    wire   [31:0] out_bits_1_fu_270_p3;
    reg   [31:0] out_bits_1_2_fu_66;
    reg   [31:0] out_bits_2_2_fu_70;
    wire    ap_block_pp0_stage0_01001;
    wire   [5:0] shl_ln_fu_154_p3;
    wire   [5:0] sub_ln403_fu_168_p2;
    wire   [5:0] sub_ln403_1_fu_180_p2;
    wire   [5:0] sub_ln403_3_fu_192_p2;
    wire   [5:0] select_ln403_fu_198_p3;
    reg   [62:0] tmp_fu_217_p4;
    wire   [5:0] select_ln403_2_fu_232_p3;
    wire   [62:0] select_ln403_1_fu_226_p3;
    wire   [62:0] zext_ln403_fu_237_p1;
    wire   [62:0] lshr_ln403_fu_244_p2;
    wire   [62:0] zext_ln403_1_fu_241_p1;
    wire   [62:0] lshr_ln403_1_fu_254_p2;
    wire   [15:0] trunc_ln403_fu_250_p1;
    wire   [15:0] trunc_ln403_1_fu_260_p1;
    wire   [15:0] and_ln403_fu_264_p2;
    reg    ap_done_reg;
    wire    ap_continue_int;
    reg    ap_done_int;
    reg   [0:0] ap_NS_fsm;
    wire    ap_enable_pp0;
    wire    ap_start_int;
    reg    ap_condition_250;
    wire    ap_ce_reg;

    // power-on initialization
    initial begin
        #0 ap_CS_fsm = 1'd1;
        #0 ap_enable_reg_pp0_iter1 = 1'b0;
        #0 i_fu_58 = 2'd0;
        #0 out_bits_0_21_fu_62 = 32'd0;
        #0 out_bits_1_2_fu_66 = 32'd0;
        #0 out_bits_2_2_fu_70 = 32'd0;
        #0 ap_done_reg = 1'b0;
    end

    main_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U (
        .ap_clk(ap_clk),
        .ap_rst(ap_rst),
        .ap_start(ap_start),
        .ap_ready(ap_ready),
        .ap_done(ap_done),
        .ap_start_int(ap_start_int),
        .ap_loop_init(ap_loop_init),
        .ap_ready_int(ap_ready_int),
        .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage0),
        .ap_loop_exit_done(ap_done_int),
        .ap_continue_int(ap_continue_int),
        .ap_done_int(ap_done_int)
    );

    always @(posedge ap_clk) begin
        if (ap_rst == 1'b1) begin
            ap_CS_fsm <= ap_ST_fsm_pp0_stage0;
        end else begin
            ap_CS_fsm <= ap_NS_fsm;
        end
    end

    always @(posedge ap_clk) begin
        if (ap_rst == 1'b1) begin
            ap_done_reg <= 1'b0;
        end else begin
            if ((ap_continue_int == 1'b1)) begin
                ap_done_reg <= 1'b0;
            end else if (((ap_loop_exit_ready == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
                ap_done_reg <= 1'b1;
            end
        end
    end

    always @(posedge ap_clk) begin
        if (ap_rst == 1'b1) begin
            ap_enable_reg_pp0_iter1 <= 1'b0;
        end else begin
            if ((1'b1 == ap_condition_exit_pp0_iter0_stage0)) begin
                ap_enable_reg_pp0_iter1 <= 1'b0;
            end else if (((1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
                ap_enable_reg_pp0_iter1 <= ap_start_int;
            end
        end
    end

    always @(posedge ap_clk) begin
        if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            if (((icmp_ln401_fu_142_p2 == 1'd0) & (ap_enable_reg_pp0_iter0 == 1'b1))) begin
                i_fu_58 <= i_3_fu_148_p2;
            end else if ((ap_loop_init == 1'b1)) begin
                i_fu_58 <= 2'd0;
            end
        end
    end

    always @(posedge ap_clk) begin
        if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            if ((ap_loop_init == 1'b1)) begin
                out_bits_0_21_fu_62 <= out_bits_0_1_reload;
            end else if (((i_2_reg_339 == 2'd0) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
                out_bits_0_21_fu_62 <= out_bits_1_fu_270_p3;
            end
        end
    end

    always @(posedge ap_clk) begin
        if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            if ((ap_loop_init == 1'b1)) begin
                out_bits_1_2_fu_66 <= out_bits_1_1_reload;
            end else if (((i_2_reg_339 == 2'd1) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
                out_bits_1_2_fu_66 <= out_bits_1_fu_270_p3;
            end
        end
    end

    always @(posedge ap_clk) begin
        if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            if ((ap_loop_init == 1'b1)) begin
                out_bits_2_2_fu_70 <= out_bits_2_1_reload;
            end else if ((1'b1 == ap_condition_250)) begin
                out_bits_2_2_fu_70 <= out_bits_1_fu_270_p3;
            end
        end
    end

    always @(posedge ap_clk) begin
        if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            i_2_reg_339 <= ap_sig_allocacmp_i_2;
            icmp_ln403_reg_351 <= icmp_ln403_fu_174_p2;
            sub_ln403_2_reg_357[5 : 4] <= sub_ln403_2_fu_186_p2[5 : 4];
            sub_ln403_4_reg_362[5 : 1] <= sub_ln403_4_fu_206_p2[5 : 1];
            sub_ln404_reg_346[5 : 4] <= sub_ln404_fu_162_p2[5 : 4];
        end
    end

    always @(*) begin
        if (((icmp_ln401_fu_142_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_subdone) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            ap_condition_exit_pp0_iter0_stage0 = 1'b1;
        end else begin
            ap_condition_exit_pp0_iter0_stage0 = 1'b0;
        end
    end

    always @(*) begin
        if (((ap_loop_exit_ready == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            ap_done_int = 1'b1;
        end else begin
            ap_done_int = ap_done_reg;
        end
    end

    always @(*) begin
        if (((ap_idle_pp0 == 1'b1) & (ap_start_int == 1'b0) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            ap_idle = 1'b1;
        end else begin
            ap_idle = 1'b0;
        end
    end

    always @(*) begin
        if (((ap_enable_reg_pp0_iter1 == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b0))) begin
            ap_idle_pp0 = 1'b1;
        end else begin
            ap_idle_pp0 = 1'b0;
        end
    end

    always @(*) begin
        if (((1'b0 == ap_block_pp0_stage0_subdone) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            ap_ready_int = 1'b1;
        end else begin
            ap_ready_int = 1'b0;
        end
    end

    always @(*) begin
        if (((ap_loop_init == 1'b1) & (1'b0 == ap_block_pp0_stage0) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            ap_sig_allocacmp_i_2 = 2'd0;
        end else begin
            ap_sig_allocacmp_i_2 = i_fu_58;
        end
    end

    always @(*) begin
        if (((icmp_ln401_fu_142_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            out_bits_0_21_out_ap_vld = 1'b1;
        end else begin
            out_bits_0_21_out_ap_vld = 1'b0;
        end
    end

    always @(*) begin
        if (((icmp_ln401_fu_142_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            out_bits_1_2_out_ap_vld = 1'b1;
        end else begin
            out_bits_1_2_out_ap_vld = 1'b0;
        end
    end

    always @(*) begin
        if (((icmp_ln401_fu_142_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            out_bits_2_2_out_ap_vld = 1'b1;
        end else begin
            out_bits_2_2_out_ap_vld = 1'b0;
        end
    end

    always @(*) begin
        case (ap_CS_fsm)
            ap_ST_fsm_pp0_stage0: begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end
            default: begin
                ap_NS_fsm = 'bx;
            end
        endcase
    end

    assign and_ln403_fu_264_p2 = (trunc_ln403_fu_250_p1 & trunc_ln403_1_fu_260_p1);

    assign ap_CS_fsm_pp0_stage0 = ap_CS_fsm[32'd0];

    assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

    assign ap_block_pp0_stage0_01001 = ~(1'b1 == 1'b1);

    assign ap_block_pp0_stage0_11001 = ~(1'b1 == 1'b1);

    assign ap_block_pp0_stage0_subdone = ~(1'b1 == 1'b1);

    always @(*) begin
        ap_condition_250 = (~(i_2_reg_339 == 2'd0) & ~(i_2_reg_339 == 2'd1) & (ap_enable_reg_pp0_iter1 == 1'b1));
    end

    assign ap_enable_pp0 = (ap_idle_pp0 ^ 1'b1);

    assign ap_enable_reg_pp0_iter0 = ap_start_int;

    assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

    assign i_3_fu_148_p2 = (ap_sig_allocacmp_i_2 + 2'd1);

    assign icmp_ln401_fu_142_p2 = ((ap_sig_allocacmp_i_2 == 2'd3) ? 1'b1 : 1'b0);

    assign icmp_ln403_fu_174_p2 = ((sub_ln404_fu_162_p2 > sub_ln403_fu_168_p2) ? 1'b1 : 1'b0);

    assign lshr_ln403_1_fu_254_p2 = 63'd9223372036854775807 >> zext_ln403_1_fu_241_p1;

    assign lshr_ln403_fu_244_p2 = select_ln403_1_fu_226_p3 >> zext_ln403_fu_237_p1;

    assign out_bits_0_21_out = out_bits_0_21_fu_62;

    assign out_bits_1_2_out = out_bits_1_2_fu_66;

    assign out_bits_1_fu_270_p3 = {{and_ln403_fu_264_p2}, {16'd32768}};

    assign out_bits_2_2_out = out_bits_2_2_fu_70;

    assign select_ln403_1_fu_226_p3 = ((icmp_ln403_reg_351[0:0] == 1'b1) ? tmp_fu_217_p4 : in_val);

    assign select_ln403_2_fu_232_p3 = ((icmp_ln403_reg_351[0:0] == 1'b1) ? sub_ln403_2_reg_357 : sub_ln404_reg_346);

    assign select_ln403_fu_198_p3 = ((icmp_ln403_fu_174_p2[0:0] == 1'b1) ? sub_ln403_1_fu_180_p2 : sub_ln403_3_fu_192_p2);

    assign shl_ln_fu_154_p3 = {{ap_sig_allocacmp_i_2}, {4'd0}};

    assign sub_ln403_1_fu_180_p2 = (sub_ln404_fu_162_p2 - sub_ln403_fu_168_p2);

    assign sub_ln403_2_fu_186_p2 = ($signed(6'd62) - $signed(sub_ln404_fu_162_p2));

    assign sub_ln403_3_fu_192_p2 = (sub_ln403_fu_168_p2 - sub_ln404_fu_162_p2);

    assign sub_ln403_4_fu_206_p2 = ($signed(6'd62) - $signed(select_ln403_fu_198_p3));

    assign sub_ln403_fu_168_p2 = ($signed(6'd62) - $signed(shl_ln_fu_154_p3));

    assign sub_ln404_fu_162_p2 = ($signed(6'd47) - $signed(shl_ln_fu_154_p3));

    integer ap_tvar_int_0;

    always @(in_val) begin
        for (ap_tvar_int_0 = 63 - 1; ap_tvar_int_0 >= 0; ap_tvar_int_0 = ap_tvar_int_0 - 1) begin
            if (ap_tvar_int_0 > 62 - 0) begin
                tmp_fu_217_p4[ap_tvar_int_0] = 1'b0;
            end else begin
                tmp_fu_217_p4[ap_tvar_int_0] = in_val[62-ap_tvar_int_0];
            end
        end
    end

    assign trunc_ln403_1_fu_260_p1 = lshr_ln403_1_fu_254_p2[15:0];

    assign trunc_ln403_fu_250_p1 = lshr_ln403_fu_244_p2[15:0];

    assign zext_ln403_1_fu_241_p1 = sub_ln403_4_reg_362;

    assign zext_ln403_fu_237_p1 = select_ln403_2_fu_232_p3;

    always @(posedge ap_clk) begin
        sub_ln404_reg_346[3:0]   <= 4'b1111;
        sub_ln403_2_reg_357[3:0] <= 4'b1111;
        sub_ln403_4_reg_362[0]   <= 1'b1;
    end

endmodule  //main_scaled_fixed2ieee_63_1_Pipeline_2