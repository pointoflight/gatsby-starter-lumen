// @flow strict
import React from 'react';
import { getContactHref } from '../../../utils';
import styles from './Author.module.scss';
import { useSiteMetadata } from '../../../hooks';

const Author = () => {
  const { author } = useSiteMetadata();

  return (
    <div className={styles['author']}>
      <p className={styles['author__bio']}>
        <a
          className={styles['author__bio-twitter']}
          href="/"
          rel="noopener noreferrer"
          target="_blank"
        >
        <strong>{author.name}</strong>
        </a>
        {author.bio}
      </p>
    </div>
  );
};

export default Author;
