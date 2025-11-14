// src/utils/constants.js

export const ROLE_CATEGORIES = [
  { value: '', label: 'All Roles' },
  { value: 'data_scientist', label: 'Data Scientist' },
  { value: 'fullstack_engineer', label: 'Full Stack Engineer' },
  { value: 'it', label: 'IT' },
];

export const RECOMMENDATION_COLORS = {
  strongly_recommend: 'bg-green-100 text-green-800 border-green-300',
  recommend: 'bg-blue-100 text-blue-800 border-blue-300',
  maybe: 'bg-yellow-100 text-yellow-800 border-yellow-300',
  not_recommended: 'bg-red-100 text-red-800 border-red-300',
};

export const RECOMMENDATION_LABELS = {
  strongly_recommend: 'Strongly Recommended',
  recommend: 'Recommended',
  maybe: 'Maybe',
  not_recommended: 'Not Recommended',
};
