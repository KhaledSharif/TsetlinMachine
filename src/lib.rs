extern crate rand;
use rand::Rng;
use rand::ThreadRng;
use rand::distributions::Uniform;
use std::cmp::min;

pub struct TsetlinMachine
{
    input_states  : Vec<bool>,
    output_states : Vec<bool>,
    outputs       : Vec<Output>,
}

impl TsetlinMachine
{
    pub fn new() -> TsetlinMachine
    {
        TsetlinMachine
        {
            input_states  : Vec::new(),
            output_states : Vec::new(),
            outputs       : Vec::new(),
        }
    }

    fn inclusion_update(&mut self, oi : usize, ci : usize, ai : usize)
    {
        let inclusion = self.outputs[oi].clauses[ci].automata_states[ai] > 0;
        let it = self.outputs[oi].clauses[ci].inclusions.iter().position(|&s| s == ai);
        if inclusion
        {
            if it.is_none()
            {
                self.outputs[oi].clauses[ci].inclusions.push(ai);
            }
        }
        else
        {
            if it.is_some()
            {
                self.outputs[oi].clauses[ci].inclusions.remove(it.unwrap());
            }
        }
    }

    fn modify_phase_one(&mut self, oi : usize, ci : usize, s_inverse : f32, s_inverse_conjugate : f32, rng : &mut ThreadRng)
    {
        let clause_state = self.outputs[oi].clauses[ci].state;
        for ai in 0..self.outputs[oi].clauses[ci].automata_states.len()
        {
            let input = if ai >= self.input_states.len() {!self.input_states[ai - self.input_states.len()]} else {self.input_states[ai]};
            let inclusion = self.outputs[oi].clauses[ci].automata_states[ai] > 0;
            let s : f32 = rng.sample(Uniform);
            if clause_state
            {
                if input
                {
                    if inclusion
                    {
                        if s < s_inverse_conjugate
                        {
                            self.outputs[oi].clauses[ci].automata_states[ai] += 1;
                            self.inclusion_update(oi, ci, ai);
                        }
                    }
                    else
                    {
                        if s < s_inverse_conjugate
                        {
                            self.outputs[oi].clauses[ci].automata_states[ai] += 1;
                            self.inclusion_update(oi, ci, ai);
                        }
                    }
                }
                else
                {
                    if !inclusion && s < s_inverse
                    {
                        self.outputs[oi].clauses[ci].automata_states[ai] -= 1;
                        self.inclusion_update(oi, ci, ai);
                    }
                }
            }
            else
            {
                if input
                {
                    if inclusion
                    {
                        if s < s_inverse
                        {
                            self.outputs[oi].clauses[ci].automata_states[ai] -= 1;
                            self.inclusion_update(oi, ci, ai);
                        }
                    }
                    else
                    {
                        if s < s_inverse
                        {
                            self.outputs[oi].clauses[ci].automata_states[ai] -= 1;
                            self.inclusion_update(oi, ci, ai);
                        }
                    }
                }
                else {
                    if inclusion
                    {
                        if s < s_inverse
                        {
                            self.outputs[oi].clauses[ci].automata_states[ai] -= 1;
                            self.inclusion_update(oi, ci, ai);
                        }
                    }
                    else
                    {
                        if s < s_inverse
                        {
                            self.outputs[oi].clauses[ci].automata_states[ai] -= 1;
                            self.inclusion_update(oi, ci, ai);
                        }
                    }
                }
            }
        }
    }

    fn modify_phase_two(&mut self, oi : usize, ci : usize)
    {
        let clause_state = self.outputs[oi].clauses[ci].state;
        for ai in 0..self.outputs[oi].clauses[ci].automata_states.len()
        {
            let input = if ai >= self.input_states.len() {!self.input_states[ai - self.input_states.len()]} else {self.input_states[ai]};
            let inclusion = self.outputs[oi].clauses[ci].automata_states[ai] > 0;
            if clause_state && !input && !inclusion
            {
                self.outputs[oi].clauses[ci].automata_states[ai] += 1;
                self.inclusion_update(oi, ci, ai);
            }
        }
    }

    pub fn create(&mut self, number_of_inputs : usize, number_of_outputs : usize, clauses_per_output : usize)
    {
        self.input_states.resize(number_of_inputs, false);
        self.outputs.resize(number_of_outputs, create_null_output());
        for oi in 0..number_of_outputs
        {
            self.outputs[oi].clauses.resize(clauses_per_output, create_null_clause());
            for ci in 0..clauses_per_output
            {
                self.outputs[oi].clauses[ci].automata_states.resize(number_of_inputs * (2 as usize), 0);
            }
        }
        self.output_states.resize(number_of_outputs, false);
    }

    pub fn learn(&mut self, target_output_states : &Vec<bool>, s : f32, t : f32, rng : &mut ThreadRng)
    {
        let s_inv = 1.0 / s;
        let s_inv_conj = 1.0 - s_inv;
        for oi in 0..self.outputs.len()
        {
            let clamped_sum = t.min((-t).max(self.outputs[oi].sum as f32));
            let rescale = 1.0 / ((2.0 * t) as f32);
            let probability_feedback_alpha = (t - clamped_sum) * rescale;
            let probability_feedback_beta  = (t + clamped_sum) * rescale;

            for ci in 0..self.outputs[oi].clauses.len()
            {
                let s : f32 = rng.sample(Uniform);
                if ci % 2 == 0
                {
                    if target_output_states[oi]
                    {
                        if s < probability_feedback_alpha
                        {
                            self.modify_phase_one(oi, ci, s_inv, s_inv_conj, rng);
                        }
                    }
                    else if s < probability_feedback_beta
                    {
                        self.modify_phase_two(oi, ci);
                    }
                }
                else
                {
                    if target_output_states[oi]
                    {
                        if s < probability_feedback_alpha
                        {
                            self.modify_phase_two(oi, ci);
                        }
                    }
                    else if s < probability_feedback_beta
                    {
                        self.modify_phase_one(oi, ci, s_inv, s_inv_conj, rng);
                    }
                }
            }
        }
    }

    pub fn activate(&mut self, input_states : Vec<bool>) -> &Vec<bool>
    {   
        self.input_states = input_states;
        for (outputs_index, mut outputs_element) in self.outputs.iter_mut().enumerate()
        {
            let mut sum = 0;
            for (clauses_index, clauses_element) in outputs_element.clauses.iter_mut().enumerate()
            {
                let mut state = true;
                for cit in clauses_element.inclusions.iter()
                {
                    let ai = *cit;
                    if ai >= self.input_states.len()
                    {
                        state = state && !self.input_states[min(self.input_states.len() - 1, ai - self.input_states.len())];
                    }
                    else
                    {
                        state = state && self.input_states[ai];
                    }
                }
                clauses_element.state = state;
                {
                    let _state = if state {1} else {0};
                    sum += if clauses_index % 2 == 0 {_state} else {-_state};
                }
            }
            outputs_element.sum = sum;
            self.output_states[outputs_index] = sum > 0;
        }
        &self.output_states
    }
}

struct Clause
{
    automata_states : Vec<i32>,
    inclusions      : Vec<usize>,
    state           : bool,
}

struct Output
{
    clauses : Vec<Clause>,
    sum     : i32,
}

fn create_null_output() -> Output
{
    Output
    {
        clauses: create_null_clauses_vector(),
        sum: 0,
    }
}

fn create_null_clauses_vector() -> Vec<Clause>
{
    Vec::new()
}

fn create_null_clause() -> Clause
{
    Clause
    {
        automata_states: Vec::new(),
        inclusions: Vec::new(),
        state: false,
    }
}

impl Clone for Output
{
    fn clone(&self) -> Output
    {
        let c = &self.clauses;
        Output
        {
            clauses: c.to_vec(),
            sum: self.sum,
        }
    }
}

impl Clone for Clause
{
    fn clone(&self) -> Clause
    {
        let a = &self.automata_states;
        let i = &self.inclusions;
        Clause
        {
            automata_states: a.to_vec(),
            inclusions: i.to_vec(),
            state: self.state,
        }
    }
}
